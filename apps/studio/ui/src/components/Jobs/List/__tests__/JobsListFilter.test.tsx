// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
    JobType,
    ListJobsQuery,
    StatedbConfigCommonJobStatus as JobStatus
} from "gen/v0/types";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setTestModelTagsResponseType } from "pages/Jobs/__tests__/MockFunctions/ModelTags.mock";
import { setTestPredictionTagsResponseType } from "pages/Jobs/__tests__/MockFunctions/PredictionTags.mock";

import JobsListFilter, { JobsListFilterProps } from "../JobsListFilter";

const setQuery = vi.fn();
const defaultProps = {
    query: {
        project_id: "project_00000000000000000000000000000001",
        limit: 10,
        page: 0,
        search_keyword: "",
        status: [],
        tags_filter: [],
        job_type: []
    } as ListJobsQuery,
    setQuery,
    projectId: "project-test",
    type: "Models",
    jobTypeOptions: [{ value: JobType.TRAINING_JOB, label: "Training" }],
    customColumns: {
        task_type: false,
        source: false
    },
    onToggleColumn: vi.fn(),
    onResetColumns: vi.fn()
};

describe("JobsListFilter", () => {
    const renderJobsListFilter = async (props: JobsListFilterProps) => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <JobsListFilter {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const predictionJobTypeOptions = [
        {
            value: JobType.BATCH_PREDICTION_JOB,
            label: "Batch Prediction"
        },
        {
            value: JobType.GENERATE_PREDICTION_TABLE_JOB,
            label: "Prediction Table"
        }
    ];

    const getPredictionProps = () => ({
        ...defaultProps,
        type: "Predictions" as const,
        jobTypeOptions: predictionJobTypeOptions
    });

    const clickDropdownOption = async (
        dropdownText: string,
        optionText: string
    ) => {
        const dropdown = await screen.findByRole("button", {
            name: dropdownText
        });
        await userEvent.click(dropdown);
        const option = await screen.findByText(optionText);
        await userEvent.click(option);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
    });

    it("Should render and call training tags API", async () => {
        await renderJobsListFilter(defaultProps);

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v0/jobs/training_tags",
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
            expect(
                screen.getByRole("button", { name: "Tags" })
            ).toBeInTheDocument();
            expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();
        });
    });

    it("Should navigate when clicking `New model` button", async () => {
        await renderJobsListFilter(defaultProps);

        const button = screen.getByText("New model");
        await userEvent.click(button);

        expect(mockUseNavigate).toHaveBeenCalledWith(
            "/project/project-test/newjob/training/graph-select/"
        );
    });

    it("Should reset filters when reset button clicked", async () => {
        await renderJobsListFilter({
            ...defaultProps,
            query: {
                project_id: "project_00000000000000000000000000000001",
                limit: 10,
                page: 0,
                search_keyword: "",
                status: ["DONE"],
                tags_filter: [],
                job_type: [JobType.TRAINING_JOB]
            } as ListJobsQuery
        });

        const resetButton = screen.getByText("Reset filters");
        await userEvent.click(resetButton);

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({
                search_keyword: undefined,
                status: [],
                sort_by: undefined,
                sort_desc: undefined,
                tags_filter: [],
                job_type: [JobType.TRAINING_JOB],
                page: 1
            })
        );
    });

    it("Should render and call prediction tags API", async () => {
        await renderJobsListFilter(getPredictionProps());

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v0/jobs/prediction_tags",
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
            expect(
                screen.getByRole("button", { name: "Tags" })
            ).toBeInTheDocument();
            expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();
        });
    });

    it("Should navigate when clicking `Generate predictions` button", async () => {
        await renderJobsListFilter(getPredictionProps());

        const button = screen.getByText("Generate predictions");
        await userEvent.click(button);

        expect(mockUseNavigate).toHaveBeenCalledWith(
            "/project/project-test/newjob/prediction/"
        );
    });

    it("Should update query when status filter changes", async () => {
        await renderJobsListFilter(defaultProps);

        await clickDropdownOption("Status", "Running");

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({ status: ["RUNNING"], page: 1 })
        );
    });

    it("Should update query when tags filter changes", async () => {
        await renderJobsListFilter(defaultProps);

        await clickDropdownOption("Tags", "prod");

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({
                tags_filter: ["prod"],
                page: 1
            })
        );
    });

    it("Should update search value and call setQuery after debounce", async () => {
        await renderJobsListFilter(defaultProps);

        const input = await screen.findByPlaceholderText("Search");
        await userEvent.type(input, "abc");

        await waitFor(
            () => {
                expect(defaultProps.setQuery).toHaveBeenCalledWith(
                    expect.objectContaining({ search_keyword: "abc", page: 1 })
                );
            },
            { timeout: 2000 }
        );
    });

    it("Should hides tags dropdown on training tag API error", async () => {
        setTestModelTagsResponseType(testResponseType.ERROR);
        await renderJobsListFilter(defaultProps);

        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            "v0/jobs/training_tags",
            undefined,
            { headers: undefined, params: { project_id: "project-tests" } }
        );
        expect(
            screen.queryByRole("button", { name: "Tags" })
        ).not.toBeInTheDocument();

        setTestModelTagsResponseType(testResponseType.ALL);
    });

    it("Should hides tags dropdown on prediction tag API error", async () => {
        setTestPredictionTagsResponseType(testResponseType.ERROR);
        await renderJobsListFilter(getPredictionProps());

        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            "v0/jobs/prediction_tags",
            undefined,
            { headers: undefined, params: { project_id: "project-tests" } }
        );
        expect(
            screen.queryByRole("button", { name: "Tags" })
        ).not.toBeInTheDocument();

        setTestPredictionTagsResponseType(testResponseType.ALL);
    });

    it("Should reset `Status` when cleared", async () => {
        await renderJobsListFilter(defaultProps);

        const statusDropdown = screen.getByText("Status");
        await userEvent.click(statusDropdown);

        const running = await screen.findByText("Running");
        await userEvent.click(running); // select Running
        await userEvent.click(running); // deselect Running

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({ status: [], page: 1 })
        );
    });

    it("Should reset `Tags` when cleared ([])", async () => {
        await renderJobsListFilter(defaultProps);

        const tagsDropdown = await screen.findByText("Tags");
        await userEvent.click(tagsDropdown);

        const tagOption = await screen.findByText("prod");
        await userEvent.click(tagOption); // select
        await userEvent.click(tagOption); // deselect

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({ tags_filter: [], page: 1 })
        );
    });

    it("Should reset tags when 'Any' is selected again", async () => {
        await renderJobsListFilter(defaultProps);

        await clickDropdownOption("Tags", "prod");

        const tagsDropdown = await screen.findByRole("button", {
            name: "Tags"
        });
        await userEvent.click(tagsDropdown);
        const anyOption = await screen.findByText("Tags");
        await userEvent.click(anyOption);

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({
                tags_filter: [],
                page: 1
            })
        );
    });

    it("Should update tags when multiple valid items are selected", async () => {
        await renderJobsListFilter(defaultProps);

        await clickDropdownOption("Tags", "prod");

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({
                tags_filter: ["prod"],
                page: 1
            })
        );

        const secondTag = await screen.findByText("growth");
        await userEvent.click(secondTag);

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({
                tags_filter: ["growth"],
                page: 1
            })
        );
    });

    it("Should toggle 'Show Archived' checkbox when dropdown menu item clicked", async () => {
        await renderJobsListFilter(defaultProps);

        const filterButton = screen.getByRole("button", { name: "Filter" });
        await userEvent.click(filterButton);

        const archivedCheckbox = screen.getByText("Show Archived");
        await userEvent.click(archivedCheckbox);

        expect(defaultProps.setQuery).toHaveBeenCalledWith(
            expect.objectContaining({
                tags_filter: ["archived"],
                page: 1
            })
        );
    });

    it("Should handle clearing status selection by clicking the selected status", async () => {
        const propsWithStatus = {
            ...defaultProps,
            setQuery,
            query: {
                ...defaultProps.query,
                status: [JobStatus.RUNNING] as JobStatus[]
            }
        };

        await renderJobsListFilter(propsWithStatus);

        const statusDropdown = screen.getByText("Status");
        await userEvent.click(statusDropdown);

        const runningOption = await screen.findByText("Running");
        await userEvent.click(runningOption);

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith(
                expect.objectContaining({
                    status: [],
                    page: 1
                })
            );
        });
    });

    it.each([
        {
            type: "Models" as const,
            getProps: () => defaultProps,
            placeholder: "Search",
            searchText: "test model"
        },
        {
            type: "Predictions" as const,
            getProps: getPredictionProps,
            placeholder: "Search",
            searchText: "test prediction"
        }
    ])(
        "Should update search value and reflect in input for $type",
        async ({ getProps, placeholder, searchText }) => {
            await renderJobsListFilter(getProps());

            const input = (await screen.findByPlaceholderText(
                placeholder
            )) as HTMLInputElement;
            await userEvent.type(input, searchText);

            expect(input.value).toBe(searchText);
        }
    );

    it("Should handle deselecting archived tag when archived is shown", async () => {
        const propsWithArchived = {
            ...defaultProps,
            setQuery,
            query: {
                ...defaultProps.query,
                tags_filter: ["archived"]
            }
        };

        await renderJobsListFilter(propsWithArchived);

        const filterButton = screen.getByRole("button", { name: "Filter 1" });
        await userEvent.click(filterButton);

        const archivedCheckbox = screen.getByText("Show Archived");
        await userEvent.click(archivedCheckbox);

        await waitFor(() => {
            expect(setQuery).toHaveBeenCalledWith(
                expect.objectContaining({
                    tags_filter: [],
                    page: 1
                })
            );
        });
    });

    it("Should display status count correctly when multiple statuses selected", async () => {
        const propsWithMultipleStatuses = {
            ...defaultProps,
            query: {
                ...defaultProps.query,
                status: [JobStatus.RUNNING, JobStatus.DONE] as JobStatus[]
            }
        };

        await renderJobsListFilter(propsWithMultipleStatuses);

        const statusButton = screen.getByRole("button", {
            name: "Status 2"
        });
        expect(statusButton).toBeInTheDocument();
    });

    it("Should clear search input when reset filters is clicked", async () => {
        await renderJobsListFilter(defaultProps);

        const input = (await screen.findByPlaceholderText(
            "Search"
        )) as HTMLInputElement;
        await userEvent.type(input, "search term");

        const resetButton = screen.getByText("Reset filters");
        await userEvent.click(resetButton);

        await waitFor(() => {
            expect(input.value).toBe("");
        });
    });
});
