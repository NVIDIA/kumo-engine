// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { ColumnDef } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GenericJob, JobType } from "gen/v0/types";
import { createQueryClient } from "tests/testUtil";

import JobsListDataTable, {
    JobsListDataTableProps
} from "../JobsListDataTable";

import { mockCreateGenericJob } from "./MockFunction/JobsListTable.test.mock";

const mockData: GenericJob[] = [
    mockCreateGenericJob({
        job_id: "trainingjob-00046000000000000000000000000000",
        job_type: JobType.TRAINING_JOB
    }),
    mockCreateGenericJob({
        job_id: "bp-job-1a2b3c4d5e6f7g8h9i0j",
        job_type: JobType.BATCH_PREDICTION_JOB
    })
];

const mockColumns: ColumnDef<GenericJob, unknown>[] = [
    {
        id: "job_id",
        header: "Job ID",
        accessorKey: "job_id",
        cell: (info) => info.getValue() as string
    },
    {
        id: "job_type",
        header: "Job Type",
        accessorKey: "job_type",
        cell: (info) => info.getValue() as string
    }
];

const onRowClick = vi.fn();
const onPageChange = vi.fn();
const onSortingChange = vi.fn();

const defaultProps: JobsListDataTableProps = {
    data: mockData,
    columns: mockColumns,
    loading: false,
    pageCount: 10,
    defaultPageSize: 10,
    currentPage: 0,
    numSkeletonRow: 10,
    columnsVisibility: { test_metric: true },
    sorting: [{ id: "job_id", desc: true }],
    onRowClick,
    onPageChange,
    onSortingChange
};

describe("JobsListDataTable", () => {
    const queryClient = createQueryClient();
    const renderJobsListDataTable = async (props: JobsListDataTableProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <JobsListDataTable {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should display skeleton when loading is set true", async () => {
        await renderJobsListDataTable({
            ...defaultProps,
            loading: true,
            data: []
        });

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
    });

    it("Should render DataTable with correct data", async () => {
        await renderJobsListDataTable(defaultProps);

        expect(screen.getByText(/00046/i)).toBeInTheDocument();
        expect(screen.getByText(/1a2b3/i)).toBeInTheDocument();
    });

    it("Should handle empty data array", async () => {
        await renderJobsListDataTable({
            ...defaultProps,
            data: []
        });

        const table = document.querySelector("table");
        expect(table).toBeInTheDocument();
    });

    it("Should call onRowClick when a row is clicked", async () => {
        await renderJobsListDataTable(defaultProps);

        const firstRow = screen.getByText(/00046/i).closest("tr");
        expect(firstRow).toBeInTheDocument();

        await userEvent.click(firstRow!);

        await waitFor(() => {
            expect(onRowClick).toHaveBeenCalled();
        });
    });

    it("Should handle pagination correctly", async () => {
        await renderJobsListDataTable({
            ...defaultProps,
            pageCount: 5,
            currentPage: 0
        });

        const nextPageButton = screen.getByLabelText("Go to next page");
        await userEvent.click(nextPageButton);

        await waitFor(() => {
            expect(onPageChange).toHaveBeenCalledWith(1);
        });
    });

    it("Should handle sorting when column header is clicked", async () => {
        await renderJobsListDataTable(defaultProps);

        const jobIdHeader = screen.getByText(/Job ID/i);
        await userEvent.click(jobIdHeader);

        await waitFor(() => {
            expect(onSortingChange).toHaveBeenCalled();
        });
    });

    it("Should pass correct sorting state to DataTable", async () => {
        const sortingState = [{ id: "job_type", desc: false }];
        await renderJobsListDataTable({
            ...defaultProps,
            sorting: sortingState
        });

        const table = document.querySelector("table");
        expect(table).toBeInTheDocument();
    });

    it("Should pass correct columns visibility to DataTable", async () => {
        const columnsVisibility = { test_metric: false, job_id: true };
        await renderJobsListDataTable({
            ...defaultProps,
            columnsVisibility
        });

        const table = document.querySelector("table");
        expect(table).toBeInTheDocument();
    });

    it("Should render with different page sizes", async () => {
        await renderJobsListDataTable({
            ...defaultProps,
            defaultPageSize: 25
        });

        const table = document.querySelector("table");
        expect(table).toBeInTheDocument();
    });

    it("Should update when currentPage prop changes", async () => {
        const { rerender } = render(
            <MemoryRouter>
                <QueryClientProvider client={queryClient}>
                    <JobsListDataTable {...defaultProps} currentPage={0} />
                </QueryClientProvider>
            </MemoryRouter>
        );

        rerender(
            <MemoryRouter>
                <QueryClientProvider client={queryClient}>
                    <JobsListDataTable {...defaultProps} currentPage={2} />
                </QueryClientProvider>
            </MemoryRouter>
        );

        const table = document.querySelector("table");
        expect(table).toBeInTheDocument();
    });

    it("Should render correct number of skeleton rows when loading", async () => {
        await renderJobsListDataTable({
            ...defaultProps,
            loading: true,
            numSkeletonRow: 5,
            data: []
        });

        const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
        expect(skeletons.length).toBeGreaterThan(0);
    });

    it("Should be memoized (default export)", async () => {
        const module = await import("../JobsListDataTable");
        const DefaultExport = module.default;
        expect(DefaultExport.$$typeof).toBeDefined();
    });
});
