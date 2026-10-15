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

import { MODEL_JOB_TYPE_OPTIONS, ModelTitle } from "utils/Constants";
import { getJobsListPageSize } from "utils/jobs";

import {
    expectedJobsListResponse,
    expectedModelListTableHeader,
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

describe("ModelsListPage", () => {
    const queryClient = createQueryClient(true);
    const renderModelListPage = async (page = 1) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        `/models/?page=${page}&job_type=TRAINING_JOB`
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <JobsList
                            title={ModelTitle}
                            jobTypeOptions={MODEL_JOB_TYPE_OPTIONS}
                        />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const verifyJobsApiCall = (params: VerifyJobsApiCallParams) =>
        sharedVerifyJobsApiCall({
            default_job_type: ["TRAINING_JOB"],
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
        job_type = ["TRAINING"]
    }: PageLoadTestArg) => {
        const defaultLimit = getJobsListPageSize();
        const pageTitle = screen.getByText("Models");
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
        expect(pageTitle).toBeInTheDocument();

        if (checkSkeleton) {
            const skeleton = screen.queryByTestId("table-skeleton");
            await waitFor(() => {
                const skeletonTableRows = screen.getAllByRole("row");
                expect(skeletonTableRows.length).toBe(10);
            });
            await waitFor(() => {
                expect(skeleton).not.toBeInTheDocument();
            });
        }

        if (checkApiCall) {
            await waitFor(() => {
                verifyJobsApiCall({
                    limit: defaultLimit,
                    page: page - 1,
                    status,
                    tags_filter: tagsFilter,
                    search_keyword: undefined,
                    sort_by: undefined,
                    sort_desc: false
                });
            });
        }

        if (checkTable && expectedJobsList.length > 0 && !search_keyword) {
            const jobsListTable = screen.getByRole("table");
            const jobsListTableRows = screen.getAllByRole("row");

            expect(jobsListTable).toBeInTheDocument();
            expect(jobsListTableRows).toHaveLength(expectedJobsList.length + 1);

            expect(jobsListTableRows[0]).toHaveTextContent(
                expectedModelListTableHeader
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
                    screen.getByText("Create your first model to get started.")
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
            window.location.origin + "/models/?page=1&job_type=TRAINING_JOB";
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

    it("Should render Model List Page with navigation", async () => {
        await renderModelListPage();
        await pageLoadTest({ page: 1 });
    });

    it("Should show welcome empty state when there are no jobs", async () => {
        setJobsTestResponseType(testResponseType.EMPTY);
        await renderModelListPage();
        await pageLoadTest({
            page: 1,
            checkTable: false,
            checkSkeleton: false,
            checkEmptyTable: true,
            checkApiCall: false
        });
        expect(
            screen.getByText(/Create your first model to get started/i)
        ).toBeInTheDocument();
    });

    it("Should navigate to the empty state for an 'Internal Server Error' response", async () => {
        setJobsTestResponseType(testResponseType.ERROR);
        await renderModelListPage();
        await pageLoadTest({
            page: 1,
            checkTable: false,
            checkSkeleton: false,
            checkEmptyTable: false
        });
        const modelsHeader = screen.getByText("Models");
        expect(modelsHeader).toBeInTheDocument();
        const emptyCard = document.querySelector('[data-slot="card"]');
        expect(emptyCard).toBeNull();
    });

    it('Should show "No matches for your selected filters." message for invalid search', async () => {
        await renderModelListPage();
        await pageLoadTest({
            page: 1,
            checkApiCall: true
        });

        const searchInput = screen.getByPlaceholderText("Search");
        const searchValue = "NotAvailabelJob";
        await userEvent.type(searchInput, searchValue);

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
        await renderModelListPage();
        await pageLoadTest({
            page: 1
        });

        const searchKey = `00047`;
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
        await renderModelListPage();
        await pageLoadTest({
            page: 1
        });

        const pageNext = screen.getByTestId("pagination-next");
        await userEvent.click(pageNext);
        await pageLoadTest({
            page: 2,
            checkSkeleton: false
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
        await renderModelListPage();
        const dropdown = screen.getByRole("button", { name: "Status" });
        expect(dropdown).toBeInTheDocument();
        await userEvent.click(dropdown);

        const menu = await screen.findByRole("menu");
        const option1 = await within(menu).findByText("Queued");
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
        await renderModelListPage();
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
        await renderModelListPage();
        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false
            })
        );

        const pageTow = screen.getByTestId("pagination-3");
        await userEvent.click(pageTow);

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
        await renderModelListPage();
        await pageLoadTest({
            page: 1
        });

        // Capture the initial list of models before applying filters
        const defaultModelList = screen.getAllByRole("row");

        const searchKey = `29d5`;
        const searchInput = screen.getByPlaceholderText("Search");
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

        // Ensure the model list after reset matches the initial list
        const ModelListAfter = await screen.findAllByRole("row");
        expect(ModelListAfter).toHaveLength(defaultModelList.length);
    });

    it("Should displays Available Tags in Dropdown and Updates Selection in Table.", async () => {
        await renderModelListPage();
        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false
            })
        );
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/jobs/training_tags`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        await userEvent.click(
            await screen.findByRole("button", { name: "Tags" })
        );
        const dropdownOptions = await screen.findAllByRole("option");
        const optionTexts = dropdownOptions.map(
            (option) => option.textContent ?? ""
        );

        // User-defined tags are offered as filter options...
        expect(optionTexts.some((t) => t.includes("prod"))).toBe(true);
        expect(optionTexts.some((t) => t.includes("growth"))).toBe(true);
        // ...but system/ID tags (pquery names & ids) are excluded so they
        // don't bloat the dropdown (kumo-ai/kumo#29898).
        expect(optionTexts.some((t) => t.includes("financial_test"))).toBe(
            false
        );
        expect(optionTexts.some((t) => t.includes("pquery-"))).toBe(false);

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

    it("Should display the 'Tune Metric' column heading with a tooltip on hover", async () => {
        await renderModelListPage();
        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false
            })
        );
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/jobs/training_tags`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        const headers = screen.getAllByRole("columnheader");
        const tuneMetricHeader = headers[2];
        expect(tuneMetricHeader).toHaveTextContent("Tune Metric");
        const labelText =
            "The value of a tune metric is the score from the best experiment on the holdout set";
        const tooltipTrigger = document.querySelector(
            '[data-slot="tooltip-trigger"]'
        );
        await userEvent.hover(tooltipTrigger!);
        await waitFor(() => {
            const tooltipContent = screen.queryByRole("tooltip");
            expect(tooltipContent).toBeInTheDocument();
        });
        const tooltipContent = screen.queryByRole("tooltip", {
            name: labelText
        });
        expect(tooltipContent).toBeInTheDocument();
    });

    it("Should display numeric values in the 'Tune Metric' column, or '—' if no value is present ", async () => {
        await renderModelListPage();
        await waitFor(() =>
            pageLoadTest({
                page: 1,
                status: [],
                checkSkeleton: false
            })
        );
        const headers = screen.getAllByRole("columnheader");
        const columnIndex = headers.findIndex((header) =>
            header.textContent?.includes("Tune Metric")
        );
        const rows = screen.getAllByRole("row").slice(1);
        rows.forEach((row) => {
            const cells = row.querySelectorAll("td");
            const tuneMetricCell = cells[columnIndex];
            expect(tuneMetricCell).toBeDefined();
            const cellText = tuneMetricCell.textContent?.trim() ?? "";
            const isValid = !isNaN(Number(cellText)) || cellText === "—";
            expect(isValid).toBe(true);
        });
    });

    it("Should show archived jobs when 'Show Archived' is checked", async () => {
        await renderModelListPage();
        await pageLoadTest({
            page: 1
        });

        const archivedJob = mockJobsListResponse.jobs.find(
            (job) =>
                job.metadata?.tags?.archived === "archived" &&
                job.job_type === "TRAINING_JOB"
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
            job_type: ["TRAINING_JOB"],
            limit: getJobsListPageSize(),
            page: 1,
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

    it("Should render the correct page when navigating using the URL parameter on the Model List page", async () => {
        await renderModelListPage(1);
        await pageLoadTest({ page: 1 });

        const page1Button = screen.getByTestId("pagination-2");
        expect(page1Button).toHaveAttribute("data-active", "true");
        expect(page1Button).toHaveAttribute("aria-current", "page");

        // Clear mock
        vi.clearAllMocks();
        cleanup();

        await renderModelListPage(3);
        await pageLoadTest({ page: 3 });

        const page3Button = screen.getByTestId("pagination-4");
        expect(page3Button).toHaveAttribute("data-active", "true");
        expect(page3Button).toHaveAttribute("aria-current", "page");
    });

    describe("Model List Page: Column Sorting Functionality", () => {
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
            await renderModelListPage();
            await pageLoadTest({ page: 1 });

            // Execute optional setup (e.g., change filters)
            if (beforeSort) {
                await beforeSort();

                // If jobTypes specified, wait for filter API call to complete
                if (jobTypes) {
                    await waitFor(
                        () => {
                            const calls = mockAxiosInstance.post.mock.calls;
                            const lastCall = calls[calls.length - 1];
                            expect(lastCall).toBeDefined();
                            expect(lastCall[0]).toBe("v0/jobs/");
                            // Check array contains same elements (order doesn't matter)
                            expect(lastCall[1].job_type).toEqual(
                                expect.arrayContaining(jobTypes)
                            );
                            expect(lastCall[1].job_type.length).toBe(
                                jobTypes.length
                            );
                        },
                        { timeout: 5000 }
                    );
                }
            }

            // Wait for data to be loaded and rendered
            await waitFor(() => {
                const rows = screen.getAllByRole("row");
                expect(rows.length).toBeGreaterThan(1);
            });

            const rowsOriginal = screen.getAllByRole("row");
            const valuesOriginal = extractCellValues(rowsOriginal, columnName);

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

            let valuesAscending: (string | undefined)[] = [];
            await waitFor(
                () => {
                    const calls = mockAxiosInstance.post.mock.calls;
                    const lastCall = calls[calls.length - 1];
                    expect(lastCall).toBeDefined();
                    expect(lastCall[0]).toBe("v0/jobs/");
                    expect(lastCall[1].sort_by).toBe(sortByValue);
                    expect(lastCall[1].sort_desc).toBe(false);

                    // If jobTypes specified, verify filter is applied
                    if (jobTypes) {
                        expect(lastCall[1].job_type).toEqual(
                            expect.arrayContaining(jobTypes)
                        );
                        expect(lastCall[1].job_type.length).toBe(
                            jobTypes.length
                        );
                    }

                    const rowsAfterAscending = screen.getAllByRole("row");
                    valuesAscending = extractCellValues(
                        rowsAfterAscending,
                        columnName
                    );
                    expect(valuesAscending).not.toEqual(valuesOriginal);
                },
                { timeout: 5000, interval: 100 }
            );

            // Second click - sort descending
            await act(async () => {
                fireEvent.click(sortIcon!);
            });

            let valuesDescending: (string | undefined)[] = [];
            await waitFor(
                () => {
                    const calls = mockAxiosInstance.post.mock.calls;
                    const lastCall = calls[calls.length - 1];
                    expect(lastCall).toBeDefined();
                    expect(lastCall[0]).toBe("v0/jobs/");
                    expect(lastCall[1].sort_by).toBe(sortByValue);
                    expect(lastCall[1].sort_desc).toBe(true);

                    // If jobTypes specified, verify filter is applied
                    if (jobTypes) {
                        expect(lastCall[1].job_type).toEqual(
                            expect.arrayContaining(jobTypes)
                        );
                        expect(lastCall[1].job_type.length).toBe(
                            jobTypes.length
                        );
                    }

                    const rowsAfterDescending = screen.getAllByRole("row");
                    valuesDescending = extractCellValues(
                        rowsAfterDescending,
                        columnName
                    );
                    expect(valuesDescending).not.toEqual(valuesOriginal);
                    expect(valuesDescending).not.toEqual(valuesAscending);
                },
                { timeout: 5000, interval: 100 }
            );

            // Third click - remove sort (back to original)
            await act(async () => {
                fireEvent.click(sortIcon!);
            });

            let valuesAfterThirdClick: (string | undefined)[] = [];
            await waitFor(
                () => {
                    const calls = mockAxiosInstance.post.mock.calls;
                    const lastCall = calls[calls.length - 1];
                    expect(lastCall).toBeDefined();
                    expect(lastCall[0]).toBe("v0/jobs/");
                    expect(lastCall[1].sort_by).toBeUndefined();

                    // If jobTypes specified, verify filter is still applied (sort removed but filter remains)
                    if (jobTypes) {
                        expect(lastCall[1].job_type).toEqual(
                            expect.arrayContaining(jobTypes)
                        );
                        expect(lastCall[1].job_type.length).toBe(
                            jobTypes.length
                        );
                    }

                    const rowsAfterThirdClick = screen.getAllByRole("row");
                    valuesAfterThirdClick = extractCellValues(
                        rowsAfterThirdClick,
                        columnName
                    );
                    expect(valuesAfterThirdClick).toEqual(valuesOriginal);
                },
                { timeout: 5000, interval: 100 }
            );

            expect(valuesAfterThirdClick).not.toEqual(valuesAscending);
            expect(valuesAfterThirdClick).not.toEqual(valuesDescending);
        };

        it.each([
            { columnName: "Job ID", sortByValue: "job_id" },
            { columnName: "Status", sortByValue: "metadata.status" },
            { columnName: "Tune Metric", sortByValue: "test_metric" },
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
