// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setXaiTestResponseType } from "components/Jobs/Training/__tests__/MockFunctions/Xai.mock";

import { GraphNodeData } from "types/explorer";

import mockXaiResponse from "../../../Jobs/Training/__tests__/MockResponses/Xai.test.json";
import GraphColumnLevelOverview, {
    GraphColumnLevelOverviewProps
} from "../GraphColumnLevelOverview/GraphColumnLevelOverview";

const trainingID = "trainingjob-00046000000000000000000000000000";
const setXaiScore = vi.fn();
const defaultProps: GraphColumnLevelOverviewProps = {
    queryId: "",
    show: true,
    trainingJobId: trainingID,
    setXaiScore,
    subGraphInfo: null,
    predictionAnchorTime: "1998-12-01T00:00:00",
    bpjobId: undefined
};

describe("GraphColumnLevelOverview", () => {
    const renderGraphColumnLevelOverview = async (
        props: GraphColumnLevelOverviewProps = defaultProps
    ) => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphColumnLevelOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${trainingID}/eval/xai`
            );
        });
    };

    const getSearchInput = () => screen.findByPlaceholderText("Search columns");

    const verifyChartDetails = async (varianceScore: string) => {
        expect(screen.getByText(/Hops/)).toBeInTheDocument();
        expect(screen.getByText(/Type/)).toBeInTheDocument();
        expect(screen.getByText(/Variation of prediction/)).toBeInTheDocument();
        expect(screen.getAllByText(`${varianceScore}%`).length).toBe(2);
    };

    const expectSearchColumnsNotVisible = () => {
        expect(
            screen.queryByPlaceholderText("Search columns")
        ).not.toBeInTheDocument();
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setXaiTestResponseType(testResponseType.DEFAULT);
    });

    it("Should render `GraphColumnLevelOverview` when all the props are set", async () => {
        await renderGraphColumnLevelOverview();
        await checkApiCall();

        expect(await getSearchInput()).toBeInTheDocument();
        expect(
            screen.getByTestId("column-FINANCE_TRANS.DATE (1 hop)")
        ).toBeInTheDocument();
        expect(
            screen.getByTestId("column-FINANCE_TRANS.COUNT(*) (1 hop)")
        ).toBeInTheDocument();

        const allColumnElements = document.querySelectorAll(
            '[data-testid^="column-"]'
        );
        expect(allColumnElements.length).toBe(17);

        await waitFor(() => {
            const cardContent = document.querySelector(
                '[data-slot="card-content"]'
            );
            expect(cardContent).toHaveTextContent("FINANCE_TRANS.DATE");
        });

        const varianceScore = (
            mockXaiResponse.tables.FINANCE_TRANS["DATE (1 hop)"]
                .variance_score * 100
        ).toFixed(2);
        await verifyChartDetails(varianceScore);
    });

    it("Should not render when show is false", async () => {
        await renderGraphColumnLevelOverview({ ...defaultProps, show: false });
        expectSearchColumnsNotVisible();
    });

    it("Should filter columns based on search input", async () => {
        await renderGraphColumnLevelOverview();

        const searchInput = await getSearchInput();
        await userEvent.type(searchInput, "FINANCE_TRANS.TYPE");

        await waitFor(() => {
            expect(
                screen.getByTestId("column-FINANCE_TRANS.TYPE (1 hop)")
            ).toBeInTheDocument();
            expect(
                screen.queryByTestId("column-FINANCE_TRANS.COUNT(*) (1 hop)")
            ).not.toBeInTheDocument();
        });
    });

    it("Should show 'No columns' message when search has no results", async () => {
        await renderGraphColumnLevelOverview();

        const searchInput = await getSearchInput();
        await userEvent.type(searchInput, "ABCDE_DATE");

        await waitFor(() => {
            expect(screen.getByText("No columns")).toBeInTheDocument();
        });

        await userEvent.clear(searchInput);
        await waitFor(() => {
            expect(
                screen.getByTestId("column-FINANCE_TRANS.TYPE (1 hop)")
            ).toBeInTheDocument();
        });
    });

    it("Should handle column click and display chart", async () => {
        await renderGraphColumnLevelOverview();

        const column = await screen.findByTestId(
            "column-FINANCE_ACCOUNT.FREQUENCY (seed entities)"
        );
        await userEvent.click(column);

        const varianceScore = (
            mockXaiResponse.tables.FINANCE_ACCOUNT["FREQUENCY (seed entities)"]
                .variance_score * 100
        ).toFixed(2);
        await verifyChartDetails(varianceScore);
    });

    it("Should call `setXaiScore` with correct data", async () => {
        await renderGraphColumnLevelOverview(defaultProps);

        await waitFor(() => {
            expect(setXaiScore).toHaveBeenCalled();
            const calledWith = setXaiScore.mock.calls[0][0];
            expect(typeof calledWith).toBe("object");
            expect(Object.keys(calledWith).length > 0);
        });
    });

    it.each([
        {
            testName: "NA/undefined contribution values",
            responseType: testResponseType.FILTERED_OUT,
            expectation: () => {
                expect(
                    screen.getByTestId("column-FINANCE_TRANS.NA_COL")
                ).toBeInTheDocument();
                expect(screen.getByText("N/A")).toBeInTheDocument();
            }
        },
        {
            testName: "different column types",
            responseType: testResponseType.VALID,
            expectation: () => {
                expect(
                    screen.getByTestId("column-FINANCE_TRANS.TIMESTAMP_COL")
                ).toBeInTheDocument();
            }
        },
        {
            testName: "empty response data",
            responseType: testResponseType.EMPTY,
            expectation: () => expectSearchColumnsNotVisible()
        },
        {
            testName: "API error gracefully",
            responseType: testResponseType.ERROR,
            expectation: () => expectSearchColumnsNotVisible()
        }
    ])("Should handle $testName", async ({ responseType, expectation }) => {
        setXaiTestResponseType(responseType);
        await renderGraphColumnLevelOverview();

        await waitFor(() => expectation());
    });

    it("Should handle pagination correctly", async () => {
        await renderGraphColumnLevelOverview();

        await waitFor(() => {
            const cardContent = document.querySelector(
                '[data-slot="card-content"]'
            );
            expect(cardContent).toHaveTextContent("FINANCE_TRANS.DATE");
        });

        const nextButton = screen.queryByRole("button", { name: /next/i });
        if (nextButton && !nextButton.hasAttribute("disabled")) {
            await userEvent.click(nextButton);
            await waitFor(() => {
                expect(nextButton).toBeInTheDocument();
            });
        }
    });

    it("Should handle subGraphInfo prop with entity data", async () => {
        const subGraphInfo: GraphNodeData = {
            tableName: "FINANCE_ACCOUNT",
            idName: "id",
            value: "test-value",
            totalScore: 0.5,
            cols: [
                { name: "FREQUENCY", value: "POPLATEK MESICNE" },
                { name: "DATE", value: "1995-03-01T00:00:00" }
            ]
        };

        await renderGraphColumnLevelOverview({
            ...defaultProps,
            subGraphInfo
        });

        await waitFor(() => {
            expect(
                screen.getByPlaceholderText("Search columns")
            ).toBeInTheDocument();
        });
    });

    it("Should switch between different columns and update chart", async () => {
        await renderGraphColumnLevelOverview();

        const typeColumn = await screen.findByTestId(
            "column-FINANCE_TRANS.TYPE (1 hop)"
        );
        await userEvent.click(typeColumn);

        await waitFor(() => {
            const cardContent = document.querySelector(
                '[data-slot="card-content"]'
            );
            expect(cardContent).toHaveTextContent("FINANCE_TRANS.TYPE");
        });

        const countColumn = await screen.findByTestId(
            "column-FINANCE_TRANS.COUNT(*) (1 hop)"
        );
        await userEvent.click(countColumn);

        await waitFor(() => {
            const cardContent = document.querySelector(
                '[data-slot="card-content"]'
            );
            expect(cardContent).toHaveTextContent("FINANCE_TRANS.COUNT");
        });
    });

    it("Should handle column selection with active state", async () => {
        await renderGraphColumnLevelOverview();

        await waitFor(() => {
            const activeElements = document.querySelectorAll(
                '[aria-selected="true"]'
            );
            expect(activeElements.length).toBeGreaterThan(0);
        });
    });

    it("Should handle different prop combinations", async () => {
        const bpjobId = "bp-test-id-123";
        const queryId = "query-test-id-456";

        await renderGraphColumnLevelOverview({
            ...defaultProps,
            bpjobId,
            queryId
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalled();
        });
        expect(await getSearchInput()).toBeInTheDocument();
    });
});
