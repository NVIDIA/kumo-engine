// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import {
    clickResetFilters,
    sharedVerifyJobsApiCall,
    VerifyJobsApiCallParams
} from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setConnectorListResponseType } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";
import { setGraphsListResponseType } from "pages/Graphs/__tests__/MockFunction/GraphsList.test.mock";
import JobsList from "pages/Jobs/JobsList";

import { PREDICTION_JOB_TYPE_OPTIONS, PredictionTitle } from "utils/Constants";
import { getJobsListPageSize } from "utils/jobs";

import {
    expectedJobsListResponse,
    expectedPredictionListTableHeader,
    setJobsTestResponseType
} from "./MockFunctions/JobList.mock";
import mockJobsListResponse from "./MockResponses/JobsList.test.response.json";

interface PageLoadTestArg {
    page: number;
    status?: Array<string>;
    search_keyword?: string;
    checkSkeleton?: boolean;
    checkApiCall?: boolean;
    checkTable?: boolean;
    checkEmptyTable?: boolean;
    tagsFilter?: Array<string>;
    sort_by?: string;
    sort_desc?: boolean;
    job_type?: Array<string>;
}

describe("PredictionsListPage", () => {
    const queryClient = createQueryClient(true);
    const renderPredictionListPage = async (page = 1) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        `/predictions/?page=${page}&job_type=BATCH_PREDICTION_JOB`
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <JobsList
                            title={PredictionTitle}
                            jobTypeOptions={PREDICTION_JOB_TYPE_OPTIONS}
                        />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const verifyJobsApiCall = (params: VerifyJobsApiCallParams) =>
        sharedVerifyJobsApiCall({
            default_job_type: ["BATCH_PREDICTION_JOB"],
            ...params
        });

    const pageLoadTest = async ({
        page = 1,
        status = [],
        search_keyword = "",
        checkSkeleton = true,
        checkApiCall = true,
        checkTable = true,
        checkEmptyTable = false,
        tagsFilter = [],
        sort_by,
        sort_desc = false,
        job_type = ["BATCH_PREDICTION_JOB"]
    }: PageLoadTestArg) => {
        const defaultLimit = getJobsListPageSize();
        const pageTitle = screen.getAllByText("Predictions");
        const expectedJobsList = expectedJobsListResponse(
            page,
            defaultLimit,
            status,
            search_keyword,
            tagsFilter,
            job_type,
            sort_by,
            sort_desc
        ) as string[];
        expect(pageTitle.length).toBeGreaterThan(0);

        if (checkSkeleton) {
            const skeleton = screen.queryByTestId("table-skeleton");
            await waitFor(async () => {
                const skeletonTableRows = await screen.findAllByRole("row");
                expect(skeletonTableRows.length).toBe(
                    expectedJobsList.length + 1
                );
            });
            await waitFor(() => {
                expect(skeleton).not.toBeInTheDocument();
            });
        }

        if (checkApiCall) {
            await waitFor(() => {
                verifyJobsApiCall({
                    job_type,
                    limit: defaultLimit,
                    page: page - 1,
                    status,
                    tags_filter: tagsFilter,
                    search_keyword: undefined,
                    sort_desc,
                    sort_by
                });
            });
        }

        if (checkTable && expectedJobsList.length > 0) {
            const jobsListTable = screen.getByRole("table");
            const jobsListTableRows = screen.getAllByRole("row");

            expect(jobsListTable).toBeInTheDocument();
            expect(jobsListTableRows).toHaveLength(expectedJobsList.length + 1);

            expect(jobsListTableRows[0]).toHaveTextContent(
                expectedPredictionListTableHeader
            );

            await waitFor(() => {
                const jobsListTableRows = screen.getAllByRole("row");

                for (let i = 1; i <= expectedJobsList.length; i++) {
                    const actual = jobsListTableRows[i].textContent
                        ?.replace(/\u200C/g, "")
                        ?.replace(/—/g, "")
                        .trim();

                    const expected = expectedJobsList[i - 1]
                        ?.replace(/\u200C/g, "")
                        ?.replace(/—/g, "")
                        .trim();

                    expect(actual?.slice(0, 5)).toBe(expected?.slice(0, 5));
                }
            });
        }

        if (checkEmptyTable) {
            await waitFor(() => {
                expect(
                    screen.getByText(
                        "Train your first model to start making predictions."
                    )
                ).toBeInTheDocument();
            });
        }
    };

    beforeEach(() => {
        Object.defineProperty(window, "innerHeight", {
            writable: true,
            configurable: true,
            value: 730
        });
        window.location.href =
            window.location.origin +
            "/predictions/?page=1&job_type=BATCH_PREDICTION_JOB";
        vi.clearAllMocks();
        setJobsTestResponseType(testResponseType.DEFAULT);
        setConnectorListResponseType(testResponseType.ALL);
        setGraphsListResponseType(testResponseType.ALL);
        queryClient.clear();
        // Suppress RUNNING/QUEUED refetchInterval timer accumulation across
        // tests.
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should render Predictions List Page with navigation", async () => {
        await renderPredictionListPage();
        await pageLoadTest({ page: 1 });
    });

    it("Should show welcome empty state when there are no jobs", async () => {
        setJobsTestResponseType(testResponseType.EMPTY);
        await renderPredictionListPage();
        await pageLoadTest({
            page: 1,
            checkTable: false,
            checkSkeleton: false,
            checkEmptyTable: true
        });

        expect(screen.getByText(/Welcome! 🎉/i)).toBeInTheDocument();
    });

    it("Should navigate to the empty state for an 'Internal Server Error' response", async () => {
        setJobsTestResponseType(testResponseType.ERROR);
        await renderPredictionListPage();
        await pageLoadTest({
            page: 1,
            checkTable: false,
            checkSkeleton: false,
            checkEmptyTable: false
        });
        const modelsHeader = screen.getByText("Predictions");
        expect(modelsHeader).toBeInTheDocument();
        const emptyCard = document.querySelector('[data-slot="card"]');
        expect(emptyCard).toBeNull();
    });

    it('Should show "No matches for your selected filters." message for invalid search', async () => {
        await renderPredictionListPage();
        await pageLoadTest({
            page: 1
        });

        const searchInput = screen.getByPlaceholderText("Search");
        await userEvent.type(searchInput, "NotAvailabelJob");

        await pageLoadTest({
            page: 1,
            checkSkeleton: false,
            checkTable: false,
            checkEmptyTable: false
        });

        await waitFor(
            () => {
                expect(
                    screen.getByText("No matches for your selected filters.")
                ).toBeInTheDocument();
            },
            { timeout: 2000 }
        );
    });

    it("Should display valid jobs for Valid search", async () => {
        await renderPredictionListPage();
        await pageLoadTest({
            page: 1
        });

        const searchKey = `00042`;
        const searchInput = await screen.findByPlaceholderText("Search");
        await userEvent.type(searchInput, searchKey);

        await waitFor(
            async () => {
                const rows = await screen.findAllByRole("row");
                const dataRows = rows.slice(1); // Skip header row
                expect(dataRows.length).toBeGreaterThan(0);

                // Verify first data row contains search key
                const firstRow = dataRows[0];
                const cells = within(firstRow).getAllByRole("cell");
                expect(cells[0].textContent).toContain(searchKey);
            },
            { timeout: 2000 }
        );
    });

    it("Should handle pagination correctly", async () => {
        await renderPredictionListPage();
        await pageLoadTest({
            page: 1
        });

        const pageNext = screen.getByTestId("pagination-next");
        fireEvent.click(pageNext);
        await pageLoadTest({
            page: 2
        });

        const pageTwo = screen.getByTestId("pagination-2");
        fireEvent.click(pageTwo);
        await pageLoadTest({
            page: 2
        });

        const pagePrev = screen.getByTestId("pagination-prev");
        fireEvent.click(pagePrev);
        await pageLoadTest({
            page: 1
        });
    });

    it("Should filter jobs based on status dropdown value selected", async () => {
        await renderPredictionListPage();
        const dropdown = screen.getByRole("button", { name: "Status" });
        expect(dropdown).toBeInTheDocument();
        await userEvent.click(dropdown);

        const menu = await screen.findByRole("menu");
        const option1 = within(menu).getByText("Queued");
        await userEvent.click(option1);

        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: ["QUEUED"],
                checkSkeleton: false,
                checkTable: false
            })
        );

        const menu1 = await screen.findByRole("menu");
        const option2 = await within(menu1).findByText("Queued");
        await userEvent.click(option2);

        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false,
                checkTable: false
            })
        );

        const menu3 = await screen.findByRole("menu");
        const option3 = await within(menu3).findByText("Done");
        await userEvent.click(option3);

        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: ["DONE"],
                checkSkeleton: false,
                checkTable: false
            })
        );
    });

    it("Should handle pagination when status dropdown value selected and go to first page when value is changed", async () => {
        await renderPredictionListPage();
        const dropdown = screen.getByRole("button", {
            name: "Status"
        });
        expect(dropdown).toBeInTheDocument();
        await userEvent.click(dropdown);

        const menu = await screen.findByRole("menu");
        const option = await within(menu).findByText("Done");
        await userEvent.click(option);

        await pageLoadTest({
            page: 1,
            status: ["DONE"],
            checkSkeleton: false,
            checkTable: false
        });

        const pageTwo = screen.getByTestId("pagination-4");
        fireEvent.click(pageTwo);

        const menu1 = await screen.findByRole("menu");
        const option3 = await within(menu1).findByText("Done");
        await userEvent.click(option3);

        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false,
                checkTable: false
            })
        );
    });

    it("Should handle pagination when the 'Reset Filters' button is clicked", async () => {
        await renderPredictionListPage();

        const resetButton = screen.getByRole("button", {
            name: /Reset Filters/i
        });
        expect(resetButton).toBeInTheDocument();
        await clickResetFilters();

        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false
            })
        );

        const pageTwo = screen.getByTestId("pagination-2");
        await userEvent.click(pageTwo);

        await clickResetFilters();

        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false
            })
        );
    });

    it("Should reset the page when the Reset Filters button is clicked.", async () => {
        await renderPredictionListPage();
        await pageLoadTest({
            page: 1
        });

        // Capture the initial list of Predictions before applying filters
        const defaultPredictionlList = screen.getAllByRole("row");

        const searchKey = `29d5`;
        const searchInput = await screen.findByPlaceholderText("Search");
        await userEvent.type(searchInput, searchKey);

        await pageLoadTest({
            page: 1,
            search_keyword: searchKey,
            checkSkeleton: false
        });

        await clickResetFilters();

        await pageLoadTest({
            page: 1,
            checkSkeleton: false,
            search_keyword: undefined
        });

        // Ensure the Predictions list after reset matches the initial list
        const predictionListAfter = screen.getAllByRole("row");
        expect(predictionListAfter).toHaveLength(defaultPredictionlList.length);
    });

    it("Should displays Available Tags in Dropdown and Updates Selection in Table.", async () => {
        await renderPredictionListPage();
        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false
            })
        );

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/jobs/prediction_tags`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );

            expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
                `v0/jobs/training_tags`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
        await userEvent.click(screen.getByRole("button", { name: "Tags" }));
        const dropdownOptions = await screen.findAllByRole("option");
        const optionTexts = dropdownOptions.map(
            (option) => option.textContent ?? ""
        );

        // User-defined tags are offered as filter options...
        expect(optionTexts.some((t) => t.includes("prod"))).toBe(true);
        expect(optionTexts.some((t) => t.includes("growth"))).toBe(true);
        // ...but system/ID tags (pquery & model ids) are excluded so they
        // don't bloat the dropdown (kumo-ai/kumo#29898).
        expect(optionTexts.some((t) => t.includes("pquery-"))).toBe(false);
        expect(optionTexts.some((t) => t.includes("trainingjob-"))).toBe(false);

        // Selecting a user tag updates the table filter.
        const tagOption = dropdownOptions.find((option) =>
            option.textContent?.includes("prod")
        );
        expect(tagOption).toBeDefined();

        if (tagOption) {
            await userEvent.click(tagOption);
        }

        await pageLoadTest({
            page: 1,
            tagsFilter: ["prod"],
            checkSkeleton: false,
            checkTable: false
        });
    });

    it("Should show archived jobs when 'Show Archived' is checked", async () => {
        await renderPredictionListPage();
        await pageLoadTest({
            page: 1
        });

        const archivedJob = mockJobsListResponse.jobs.find(
            (job) =>
                job.metadata?.tags?.archived === "archived" &&
                job.job_type === "BATCH_PREDICTION_JOB"
        );
        expect(archivedJob).toBeDefined();
        const shortenedId = archivedJob!.job_id.split("-").pop()!.slice(0, 5);
        expect(screen.queryByTestId(shortenedId)).not.toBeInTheDocument();

        const filterButton = screen.getByRole("button", { name: "Filter" });
        expect(filterButton).toBeInTheDocument();
        await userEvent.click(filterButton);

        const checkbox = document.querySelector('input[type="checkbox"]');
        expect(checkbox).not.toBeChecked();
        await userEvent.click(checkbox!);
        expect(checkbox).toBeChecked();

        verifyJobsApiCall({
            job_type: ["BATCH_PREDICTION_JOB"],
            limit: getJobsListPageSize(),
            page: 0,
            status: [],
            tags_filter: ["archived"],
            search_keyword: undefined,
            sort_by: undefined,
            sort_desc: false
        });

        const jobRows = document.querySelectorAll('[data-slot="table-row"]');
        expect(jobRows).toHaveLength(2); //Along with the table header

        //Job ID check
        const archivedJobRowText = jobRows[1].textContent;
        expect(archivedJobRowText).toContain(shortenedId);
    });

    it("Should render the correct page when navigating using the URL parameter on the Prediction List page", async () => {
        await renderPredictionListPage(1);
        await pageLoadTest({ page: 1 });

        const page1Button = screen.getByTestId("pagination-2");
        expect(page1Button).toHaveAttribute("data-active", "true");
        expect(page1Button).toHaveAttribute("aria-current", "page");

        // Clear mock
        vi.clearAllMocks();
        cleanup();

        await renderPredictionListPage(3);
        await pageLoadTest({ page: 3 });

        const page3Button = screen.getByTestId("pagination-4");
        expect(page3Button).toHaveAttribute("data-active", "true");
        expect(page3Button).toHaveAttribute("aria-current", "page");
    });

    describe("Prediction List Page: Column Sorting Functionality", () => {
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
            await renderPredictionListPage();
            await pageLoadTest({ page: 1 });

            // Execute optional setup
            if (beforeSort) {
                await beforeSort();
            }

            const rowsOriginal = screen.getAllByRole("row");
            // Compare stable row identities, not the rendered sort value.
            // Relative-time labels are intentionally lossy (for example,
            // distinct timestamps can all render as "1 year ago").
            const valuesOriginal = extractCellValues(rowsOriginal, "Job ID");

            // Find the column header within the table header row to avoid conflicts with filter dropdowns
            const headerRow = rowsOriginal[0];
            const headers = Array.from(headerRow.querySelectorAll("th"));
            const columnHeader = headers.find((th) =>
                th.textContent?.includes(columnName)
            );
            const sortIcon = columnHeader?.querySelector("svg");

            await act(async () => {
                fireEvent.click(sortIcon!);
            });

            await pageLoadTest({
                page: 1,
                sort_by: sortByValue,
                sort_desc: false,
                checkSkeleton: false,
                job_type: jobTypes
            });

            const rowsAfterAscending = screen.getAllByRole("row");
            const valuesAscending = extractCellValues(
                rowsAfterAscending,
                "Job ID"
            );

            await act(async () => {
                fireEvent.click(sortIcon!);
            });

            await pageLoadTest({
                page: 1,
                sort_by: sortByValue,
                sort_desc: true,
                checkSkeleton: false,
                job_type: jobTypes
            });

            const rowsAfterDescending = screen.getAllByRole("row");
            const valuesDescending = extractCellValues(
                rowsAfterDescending,
                "Job ID"
            );

            expect(valuesDescending).not.toEqual(valuesAscending);

            await act(async () => {
                fireEvent.click(sortIcon!);
            });

            await pageLoadTest({
                page: 1,
                checkSkeleton: false,
                job_type: jobTypes
            });

            const rowsAfterThirdClick = screen.getAllByRole("row");
            const valuesAfterThirdClick = extractCellValues(
                rowsAfterThirdClick,
                "Job ID"
            );

            expect(valuesAfterThirdClick).toEqual(valuesOriginal);
        };

        it.each([
            { columnName: "Job ID", sortByValue: "job_id" },
            { columnName: "Status", sortByValue: "metadata.status" },
            { columnName: "Started", sortByValue: "metadata.start_time" },
            { columnName: "Duration", sortByValue: "duration" },
            { columnName: "Tags", sortByValue: "metadata.tags" }
        ])(
            "Should sort $columnName column on click and toggle between ascending, descending, and back to original",
            async ({ columnName, sortByValue }) => {
                await testColumnSort(columnName, sortByValue);
            }
        );
    });
});
