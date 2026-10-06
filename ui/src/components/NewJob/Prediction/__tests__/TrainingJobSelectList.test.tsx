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
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setConnectorListResponseType } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";
import { setGraphsListResponseType } from "pages/Graphs/__tests__/MockFunction/GraphsList.test.mock";
import { setJobsTestResponseType } from "pages/Jobs/__tests__/MockFunctions/JobList.mock";
import mockJobs from "../../../../pages/Jobs/__tests__/MockResponses/JobsList.test.response.json";

import TrainingJobSelectList, {
    TrainingJobSelectListProps
} from "../Select/TrainingJobSelectList";

interface PageLoadTest {
    page?: number;
    status?: Array<string>;
    search_keyword?: string;
    checkApiCall?: boolean;
    tagsFilter?: Array<string>;
    sort_by?: string;
    sort_desc?: boolean;
    job_type?: Array<string>;
}

const setNext = vi.fn();
const defaultProps: TrainingJobSelectListProps = {
    setNext,
    projectId: "project-tests"
};

describe("TrainingJobSelectList", () => {
    const queryClient = createQueryClient();
    const renderTrainingJobSelectList = async (
        props: TrainingJobSelectListProps = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <TrainingJobSelectList {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const pageLoadTest = async ({
        page = 1,
        status = ["DONE"],
        search_keyword = "",
        checkApiCall = true,
        sort_by,
        sort_desc
    }: PageLoadTest) => {
        const defaultLimit = Math.max(
            1,
            Math.floor((window.innerHeight - 285) / 44)
        );

        if (checkApiCall) {
            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v0/jobs/",
                    {
                        job_type: ["TRAINING_JOB"],
                        limit: defaultLimit,
                        page: page - 1,
                        project_id: "project-tests",
                        status,
                        tags: {},
                        time_range_end: undefined,
                        time_range_start: undefined,
                        search_keyword,
                        sort_desc,
                        sort_by
                    },
                    {
                        "headers": {
                            "Content-Type": "application/json"
                        }
                    }
                );
            });
        }
    };

    const extractCellValues = (
        rows: HTMLElement[],
        columnName: string
    ): (string | undefined)[] => {
        const headerRow = rows[0];
        const headers = Array.from(headerRow.querySelectorAll("th"));
        const columnIndex = headers.findIndex((th) =>
            th.textContent?.includes(columnName)
        );

        return rows.slice(1).map((row) => {
            const cells = row.querySelectorAll("td");
            return cells[columnIndex]?.textContent?.trim();
        });
    };

    const testColumnSort = async (
        columnName: string,
        sortByValue: string,
        beforeSort?: () => Promise<void>,
        jobTypes?: Array<string>
    ) => {
        await renderTrainingJobSelectList();
        await pageLoadTest({ page: 1 });

        // Execute optional setup
        if (beforeSort) {
            await beforeSort();
        }

        const rowsOriginal = screen.getAllByRole("row");
        const valuesOriginal = extractCellValues(rowsOriginal, columnName);

        const column = screen.getByText(columnName);
        const sortIcon = column.closest("th")?.querySelector("svg");

        await act(async () => {
            fireEvent.click(sortIcon!);
        });

        await pageLoadTest({
            page: 1,
            sort_by: sortByValue,
            sort_desc: false,
            job_type: jobTypes
        });

        const rowsAfterAscending = screen.getAllByRole("row");
        const valuesAscending = extractCellValues(
            rowsAfterAscending,
            columnName
        );

        expect(valuesAscending).not.toEqual(valuesOriginal);

        await act(async () => {
            fireEvent.click(sortIcon!);
        });

        await pageLoadTest({
            page: 1,
            sort_by: sortByValue,
            sort_desc: true,
            job_type: jobTypes
        });

        const rowsAfterDescending = screen.getAllByRole("row");
        const valuesDescending = extractCellValues(
            rowsAfterDescending,
            columnName
        );

        expect(valuesDescending).not.toEqual(valuesOriginal);
        expect(valuesDescending).not.toEqual(valuesAscending);

        await act(async () => {
            fireEvent.click(sortIcon!);
        });

        await pageLoadTest({
            page: 1,
            job_type: jobTypes
        });

        const rowsAfterThirdClick = screen.getAllByRole("row");
        const valuesAfterThirdClick = extractCellValues(
            rowsAfterThirdClick,
            columnName
        );

        expect(valuesAfterThirdClick).toEqual(valuesOriginal);
        expect(valuesAfterThirdClick).not.toEqual(valuesAscending);
        expect(valuesAfterThirdClick).not.toEqual(valuesDescending);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setJobsTestResponseType(testResponseType.ALL);
    });

    it("Should display table data when available", async () => {
        await renderTrainingJobSelectList();

        expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();
        await waitFor(() => {
            expect(
                screen.getByText(mockJobs.jobs[0].job_id.slice(12, 17))
            ).toBeInTheDocument();
        });
    });

    it("Should handle search input", async () => {
        await renderTrainingJobSelectList();

        const searchInput = await screen.findByPlaceholderText("Search");
        const input = "9f8b2";
        await userEvent.type(searchInput, input);

        await waitFor(
            () => {
                const rows = document.querySelectorAll(
                    "[data-slot='table-row']"
                );
                expect(rows.length).toBe(2); //Including table-header

                expect(screen.getByText(input)).toBeInTheDocument();
            },
            { timeout: 2000 }
        );
    });

    it("Should display `No jobs found` for invalid search input", async () => {
        await renderTrainingJobSelectList();
        const searchInput = await screen.findByPlaceholderText("Search");
        const input = "#$%32000";
        await userEvent.type(searchInput, input);

        await waitFor(
            async () => {
                expect(
                    await screen.findByText(/No jobs found/i)
                ).toBeInTheDocument();
            },
            { timeout: 2000 }
        );
    });

    it("Should handle `Job ID` click", async () => {
        await renderTrainingJobSelectList();

        const jobId = mockJobs.jobs[0].job_id;
        const trimmedId = jobId.split("-")[1].slice(0, 5);
        const row = screen.getByText(trimmedId);
        await userEvent.click(row);

        expect(setNext).toHaveBeenCalledWith(mockJobs.jobs[0]);
    });

    it("Should handle checkbox click", async () => {
        await renderTrainingJobSelectList();

        const arrowCell = await screen.findAllByRole("cell", { name: "" });
        const firstRowCell = arrowCell[0];
        await userEvent.click(firstRowCell);

        expect(setNext).toHaveBeenCalledWith(mockJobs.jobs[0]);
    });

    it("Should handle pagination and update page correctly", async () => {
        await renderTrainingJobSelectList();

        await waitFor(() => {
            expect(
                screen.getByText(mockJobs.jobs[0].job_id.slice(12, 17))
            ).toBeInTheDocument();
        });

        const pageNext = await screen.findByTestId("pagination-next");
        await userEvent.click(pageNext);

        const pageTwo = screen.queryByTestId("pagination-2");
        expect(pageTwo).toBeInTheDocument();
        await userEvent.click(pageTwo!);

        const pagePrev = screen.getByTestId("pagination-prev");
        await userEvent.click(pagePrev);

        await waitFor(() => {
            expect(
                screen.getByText(mockJobs.jobs[0].job_id.slice(12, 17))
            ).toBeInTheDocument();
        });
    });

    it("Should show tooltip content on hover over", async () => {
        await renderTrainingJobSelectList();

        const jobId = mockJobs.jobs[0].job_id;
        const trimmedId = jobId.split("-")[1].slice(0, 5);
        const jobIdElement = await screen.findByText(trimmedId);

        userEvent.hover(jobIdElement);
        await waitFor(() => {
            expect(screen.getByText("Query")).toBeInTheDocument();
        });
    });

    it("Should display welcome empty state for Empty job list", async () => {
        queryClient.clear();
        setJobsTestResponseType(testResponseType.EMPTY);
        setConnectorListResponseType(testResponseType.ALL);
        setGraphsListResponseType(testResponseType.ALL);
        await renderTrainingJobSelectList();

        await waitFor(async () => {
            expect(await screen.findByText(/Welcome! 🎉/i)).toBeInTheDocument();
        });

        expect(
            screen.getByRole("button", { name: "New model" })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "View docs" })
        ).toBeInTheDocument();
    });

    it("Should sort Job ID column on click and toggle between ascending, descending, and back to original", async () => {
        queryClient.clear();
        await testColumnSort("Job ID", "job_id");
    });
});
