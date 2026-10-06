// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GenericJob, JobType, ListJobsQuery } from "gen/v0/types";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { createQueryClient } from "tests/testUtil";

import mockGetJobsList from "pages/Jobs/__tests__/MockResponses/JobsList.test.response.json";

import { jobStatus } from "utils/Constants";

import JobsListTable, { JobsListTableProps } from "../JobsListTable";

const setQuery = vi.fn();
const defaultProps: JobsListTableProps = {
    query: {
        limit: 10,
        page: 0
    } as ListJobsQuery,
    setQuery,
    jobs: mockGetJobsList.jobs as unknown as GenericJob[],
    refetchJobs: vi.fn(),
    totalNumJobs: 1,
    loading: false,
    jobTypeOptions: []
};

describe("JobsListTable", () => {
    const queryClient = createQueryClient();
    const renderJobsListTable = async (props: JobsListTableProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <JobsListTable {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
    });

    it("Should display `skeleton` when `loading` is set true", async () => {
        await renderJobsListTable({
            ...defaultProps,
            query: {} as ListJobsQuery,
            jobs: [] as GenericJob[],
            totalNumJobs: 0,
            loading: true
        });

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
    });

    it('Should show "No matches for your selected filters." for when there are no jobs', async () => {
        await renderJobsListTable({
            ...defaultProps,
            jobs: []
        });

        expect(
            screen.getByText("No matches for your selected filters.")
        ).toBeInTheDocument();
    });

    it("Should reset filters when reset button is clicked", async () => {
        await renderJobsListTable({
            ...defaultProps,
            query: {
                ...defaultProps.query,
                search_keyword: "test",
                status: [jobStatus.Running],
                tags_filter: ["tag1"]
            },
            jobTypeOptions: [
                { value: JobType.TRAINING_JOB, label: "Training" }
            ],
            jobs: []
        });

        const resetButton = screen.getByText("Reset filters");
        await userEvent.click(resetButton);

        expect(setQuery).toHaveBeenCalledWith({
            ...defaultProps.query,
            search_keyword: undefined,
            status: [],
            tags_filter: [],
            job_type: ["TRAINING_JOB"],
            page: 1
        });
    });

    it("Should set body overflow to hidden when job overview is open", async () => {
        await renderJobsListTable(defaultProps);

        expect(document.body.style.overflow).toBe("");

        const jobRow = screen.getByText(/9f8b2/).closest("tr");
        if (jobRow) {
            fireEvent.click(jobRow);
        }

        expect(document.body.style.overflow).toBe("hidden");
        document.body.style.overflow = "";
    });

    it("Should handle pagination", async () => {
        await renderJobsListTable({
            ...defaultProps,
            totalNumJobs: 25
        });

        const tableHeader = screen.getByTestId("table-header");
        expect(tableHeader).toHaveTextContent(
            "Job IDStatusStartedUserDurationTags"
        );

        const nextPageButton = screen.getByLabelText("Go to next page");
        await userEvent.click(nextPageButton);

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                page: 2
            });
        });
    });

    it("Should show tags, job status, and the hover card for job ID", async () => {
        await renderJobsListTable(defaultProps);

        expect(screen.getByText("financial_test")).toBeInTheDocument();
        expect(screen.getAllByText(/Done/i)).not.toBeNull();

        // The Job ID hash is a Radix HoverCard trigger. The virtualized
        // DataTable remounts its cells once while the initial data settles;
        // hovering before that completes races with the remount (the trigger
        // node is swapped out, clearing the hover-open timer) and the card
        // never opens. Wait for the trigger node to stop being replaced, then
        // hover the visible hash exactly as a user would.
        let prevTrigger: Element | null = null;
        await waitFor(() => {
            const trigger = screen
                .getByText("9f8b2")
                .closest('[data-slot="hover-card-trigger"]');
            if (!trigger || trigger !== prevTrigger) {
                prevTrigger = trigger;
                throw new Error("hover-card trigger not stable yet");
            }
        });

        await userEvent.hover(screen.getByText("9f8b2"));

        await waitFor(() => {
            expect(screen.getByText("Query")).toBeInTheDocument();
        });
    });

    it("Should open job overview drawer on row click", async () => {
        await renderJobsListTable(defaultProps);

        const jobId = "trainingjob-9f8b259b4f764c8f932ad5912a926ec2";
        const displayId = jobId.replace("trainingjob-", "").slice(0, 5);
        const jobIdElement = screen.getByText(displayId);
        expect(jobIdElement).toBeInTheDocument();

        const jobRow = jobIdElement.closest("tr");
        expect(jobRow).toBeInTheDocument();
        await userEvent.click(jobRow!);

        expect(mockUseNavigate).toHaveBeenCalledWith(
            `/project/project-tests/models/training/${jobId}`,
            { replace: true }
        );
    });

    it("Should display both Training and Prediction jobs from mockJobs", async () => {
        const trainingJob = mockGetJobsList.jobs.find(
            (job) => job.job_type === "TRAINING_JOB"
        );
        const predictionJob = mockGetJobsList.jobs.find(
            (job) => job.job_type === "BATCH_PREDICTION_JOB"
        );

        expect(trainingJob).toBeDefined();
        expect(predictionJob).toBeDefined();

        await renderJobsListTable({
            ...defaultProps,
            jobs: mockGetJobsList.jobs as unknown as GenericJob[]
        });

        const trainingDisplayId = trainingJob!.job_id
            .replace("trainingjob-", "")
            .slice(0, 5);
        const predictionDisplayId = predictionJob!.job_id
            .replace("bp-job-", "")
            .slice(0, 5);

        expect(screen.getByText(trainingDisplayId)).toBeInTheDocument();
        expect(screen.getByText(predictionDisplayId)).toBeInTheDocument();

        expect(screen.getAllByText(/Training/i).length).toBeGreaterThan(0);
        expect(screen.getAllByText(/Prediction/i).length).toBeGreaterThan(0);
    });

    it("Should call setQuery with sort_by: 'job_id' for 'Job ID' sort", async () => {
        await renderJobsListTable({
            ...defaultProps,
            jobs: mockGetJobsList.jobs as unknown as GenericJob[]
        });

        //SORT ASC
        const jobIdHeader = screen.getByText(/Job ID/i);
        fireEvent.click(jobIdHeader);

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "job_id",
                sort_desc: false,
                page: 1
            });
        });

        //SORT DESC
        await userEvent.click(screen.getByText(/Job ID/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "job_id",
                sort_desc: true,
                page: 1
            });
        });

        //SORT ASC
        await userEvent.click(screen.getByText(/Job ID/i));
        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "job_id",
                sort_desc: false,
                page: 1
            });
        });
    });

    it("Should call setQuery with sort_by: 'test_metric' for 'Tune Metric' sort", async () => {
        await renderJobsListTable({
            ...defaultProps,
            query: {
                limit: 10,
                page: 0,
                job_type: ["TRAINING_JOB"]
            } as ListJobsQuery,
            jobs: mockGetJobsList.jobs as unknown as GenericJob[]
        });

        //SORT ASC
        fireEvent.click(screen.getByText(/Tune Metric/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                job_type: ["TRAINING_JOB"],
                sort_by: "test_metric",
                sort_desc: false,
                page: 1
            });
        });

        //SORT DESC
        fireEvent.click(screen.getByText(/Tune Metric/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                job_type: ["TRAINING_JOB"],
                sort_by: "test_metric",
                sort_desc: true,
                page: 1
            });
        });

        //SORT ASC
        fireEvent.click(screen.getByText(/Tune Metric/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                job_type: ["TRAINING_JOB"],
                sort_by: "test_metric",
                sort_desc: false,
                page: 1
            });
        });
    });

    it("Should call setQuery with sort_by: 'metadata.start_time' for 'Started' sort", async () => {
        await renderJobsListTable({
            ...defaultProps,
            jobs: mockGetJobsList.jobs as unknown as GenericJob[]
        });

        //SORT ASC
        const startedHeader = screen.getByText(/Started/i);
        await userEvent.click(startedHeader);

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.start_time",
                sort_desc: false,
                page: 1
            });
        });

        //SORT DESC
        await userEvent.click(screen.getByText(/Started/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.start_time",
                sort_desc: true,
                page: 1
            });
        });

        //SORT ASC
        await userEvent.click(screen.getByText(/Started/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.start_time",
                sort_desc: false,
                page: 1
            });
        });
    });

    it("Should call setQuery with sort_by: 'metadata.status' for 'Status' sort", async () => {
        await renderJobsListTable({
            ...defaultProps,
            jobs: mockGetJobsList.jobs as unknown as GenericJob[]
        });

        //SORT ASC
        const statusHeader = screen.getByText(/Status/i);
        await userEvent.click(statusHeader);

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.status",
                sort_desc: false,
                page: 1
            });
        });

        //SORT DESC
        await userEvent.click(screen.getByText(/Status/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.status",
                sort_desc: true,
                page: 1
            });
        });

        //SORT ASC
        await userEvent.click(screen.getByText(/Status/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.status",
                sort_desc: false,
                page: 1
            });
        });
    });

    it("Should call setQuery with sort_by: 'duration' for 'Duration' sort", async () => {
        await renderJobsListTable({
            ...defaultProps,
            jobs: mockGetJobsList.jobs as unknown as GenericJob[]
        });

        //SORT ASC
        const durationHeader = screen.getByText(/Duration/i);
        await userEvent.click(durationHeader);

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "duration",
                sort_desc: false,
                page: 1
            });
        });

        //SORT DESC
        await userEvent.click(screen.getByText(/Duration/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "duration",
                sort_desc: true,
                page: 1
            });
        });

        //SORT ASC
        await userEvent.click(screen.getByText(/Duration/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "duration",
                sort_desc: false,
                page: 1
            });
        });
    });

    it("Should call setQuery with sort_by: 'metadata.tags' for 'Tags' sort", async () => {
        await renderJobsListTable({
            ...defaultProps,
            jobs: mockGetJobsList.jobs as unknown as GenericJob[]
        });

        //SORT ASC
        const tagsHeader = screen.getByText(/Tags/i);
        await userEvent.click(tagsHeader);

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.tags",
                sort_desc: false,
                page: 1
            });
        });

        //SORT DESC
        await userEvent.click(screen.getByText(/Tags/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.tags",
                sort_desc: true,
                page: 1
            });
        });

        //SORT ASC
        await userEvent.click(screen.getByText(/Tags/i));

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith({
                ...defaultProps.query,
                sort_by: "metadata.tags",
                sort_desc: false,
                page: 1
            });
        });
    });
});
