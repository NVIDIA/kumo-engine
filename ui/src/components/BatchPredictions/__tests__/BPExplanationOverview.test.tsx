// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";

import { UnlinkedBatchPredictionJobData } from "types/batchPredictions";
import { TaskType } from "types/predictiveQueries";

import BPExplanationOverview, {
    BPExplanationProps
} from "../Explanation/BPExplanationOverview";

import { MockUnlinkedBatchPredictionJobData } from "./MockResponses/UnlinkedBatchPredictionJobData";

const defaultProps: BPExplanationProps = {
    jobInfo: MockUnlinkedBatchPredictionJobData,
    show: true,
    predictiveQueryType: TaskType.BINARY_CLASSIFICATION,
    queryYaml:
        "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
    showGroundTruth: true,
    jobId: MockUnlinkedBatchPredictionJobData.id
};

describe("BPExplanationOverview", () => {
    const queryClient = createQueryClient();

    const renderBPExplanationOverview = async (props: BPExplanationProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <BPExplanationOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            const APIS_TO_CALL = [
                `v1/batch_prediction_jobs/${MockUnlinkedBatchPredictionJobData.id}/available_xai_entities`,
                `v1/batch_prediction_jobs/${MockUnlinkedBatchPredictionJobData.id}/xai`
            ];
            APIS_TO_CALL.forEach((API) => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `${API}`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            });
        });
    };

    const getGlobalTab = () =>
        screen.getByRole("tab", { name: "Global column explanations" });
    const getLocalTab = () =>
        screen.getByRole("tab", { name: "Local subgraph explanation" });

    const verifyTabsExist = () => {
        expect(getGlobalTab()).toBeInTheDocument();
        expect(getLocalTab()).toBeInTheDocument();
    };

    const switchToLocalTab = async () => {
        await userEvent.click(screen.getByText("Local subgraph explanation"));
    };

    const getSearchInput = async () => {
        await waitFor(() =>
            expect(screen.getByTestId("xai-columns-search")).toBeInTheDocument()
        );
        return screen.getByPlaceholderText("Search columns");
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should render skeleton when jobInfo is empty and jobId is missing", async () => {
        await renderBPExplanationOverview({
            jobInfo: {} as unknown as UnlinkedBatchPredictionJobData,
            show: false,
            predictiveQueryType: TaskType.BINARY_CLASSIFICATION,
            queryYaml: "",
            showGroundTruth: false,
            jobId: ""
        });

        verifyTabsExist();

        const skeletonElement = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(skeletonElement).toBeInTheDocument();
    });

    it("Should display all the content under the `Global` tab", async () => {
        await renderBPExplanationOverview(defaultProps);
        await checkApiCall();

        verifyTabsExist();

        const columnName = "FINANCE_ORDERS.AMOUNT";
        await waitFor(() => {
            expect(
                screen.getByTestId(`column-${columnName} (1 hop)`)
            ).toBeInTheDocument();
            expect(screen.getAllByText(`${columnName} (1 hop)`).length).toBe(1);
            expect(screen.getByText("Hops")).toBeInTheDocument();
            expect(screen.getByText("Type")).toBeInTheDocument();
            expect(
                screen.getByText("Variation of prediction")
            ).toBeInTheDocument();
            expect(screen.getByTestId("bar-chart")).toBeInTheDocument();
        });
    });

    it("Should filter the columns based on the `Search`", async () => {
        await renderBPExplanationOverview(defaultProps);
        await checkApiCall();

        const searchedColumn = "FINANCE_TRANS.BALANCE";
        const nonSearchedColumn = "FINANCE_ORDERS.AMOUNT";

        await waitFor(() => {
            expect(
                screen.getByTestId(`column-${searchedColumn} (1 hop)`)
            ).toBeInTheDocument();
            expect(
                screen.getByTestId(`column-${nonSearchedColumn} (1 hop)`)
            ).toBeInTheDocument();
        });

        const searchInput = await getSearchInput();
        await userEvent.type(searchInput, searchedColumn);

        expect(
            screen.getByTestId(`column-${searchedColumn} (1 hop)`)
        ).toBeInTheDocument();
        expect(screen.getAllByText(`${searchedColumn} (1 hop)`).length).toBe(1);
        expect(
            screen.queryByTestId(`column-${nonSearchedColumn} (1 hop)`)
        ).not.toBeInTheDocument();
    });

    it("Should display `No columns` for non-existing `Search`", async () => {
        await renderBPExplanationOverview(defaultProps);
        await checkApiCall();

        const searchInput = await getSearchInput();
        await userEvent.type(searchInput, "302000#5");
        await waitFor(() =>
            expect(screen.getByText("No columns")).toBeInTheDocument()
        );
    });

    it("Should display Local tab content and navigate between samples using Previous/Next buttons", async () => {
        await renderBPExplanationOverview(defaultProps);
        await checkApiCall();

        await switchToLocalTab();

        verifyTabsExist();

        const previousButton = screen.getByRole("button", {
            name: /Previous/i
        });
        const nextButton = screen.getByRole("button", { name: /Next/i });
        expect(previousButton).toBeInTheDocument();
        expect(nextButton).toBeInTheDocument();
        expect(screen.getByText("Sample details")).toBeInTheDocument();

        const initialId = "26";
        expect(screen.getByText(initialId)).toBeInTheDocument();

        await userEvent.click(nextButton);
        const nextId = "29";
        expect(screen.getByText(nextId)).toBeInTheDocument();
        expect(screen.queryByText(initialId)).not.toBeInTheDocument();

        await userEvent.click(previousButton);
        expect(screen.getByText(initialId)).toBeInTheDocument();
        expect(screen.queryByText(nextId)).not.toBeInTheDocument();
    });

    it("Should display content under the Sample details", async () => {
        await renderBPExplanationOverview(defaultProps);
        await checkApiCall();

        await switchToLocalTab();

        expect(screen.getByText("Sample details")).toBeInTheDocument();
        expect(screen.getByText("Output")).toBeInTheDocument();
        expect(screen.getByText("Prediction")).toBeInTheDocument();
        expect(screen.getByText("Ground Truth")).toBeInTheDocument();
        expect(screen.getByText("Anchor Time")).toBeInTheDocument();
    });

    it("Should not display `Ground Truth` when `showGroundTruth` is set as false", async () => {
        await renderBPExplanationOverview({
            ...defaultProps,
            showGroundTruth: false
        });
        await checkApiCall();

        await switchToLocalTab();

        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
        expect(screen.getByText("Anchor Time")).toBeInTheDocument();
    });

    it.each([
        {
            taskType: TaskType.MULTILABEL_RANKING,
            taskName: "MULTILABEL RANKING"
        },
        { taskType: TaskType.LINK_PREDICTION, taskName: "LINK PREDICTION" }
    ])(
        "Should display `Column Analysis is not available` for $taskName task type",
        async ({ taskType, taskName }) => {
            await renderBPExplanationOverview({
                ...defaultProps,
                predictiveQueryType: taskType
            });

            verifyTabsExist();

            await waitFor(() => {
                expect(
                    screen.getByText(
                        new RegExp(
                            `Column Analysis is not available for ${taskName} tasks. For more information, please contact kumo support.`
                        )
                    )
                ).toBeInTheDocument();
            });
        }
    );

    it("Should maintain state when switching between Global and Local tabs", async () => {
        await renderBPExplanationOverview(defaultProps);
        await checkApiCall();

        const xaiGlobal = getGlobalTab();
        const searchInput = await getSearchInput();
        await userEvent.type(searchInput, "FINANCE");

        await userEvent.click(getLocalTab());

        await userEvent.click(xaiGlobal);
        expect(searchInput).toHaveValue("FINANCE");
    });

    it("Should display active tab styling correctly", async () => {
        await renderBPExplanationOverview(defaultProps);
        await checkApiCall();

        const xaiGlobal = getGlobalTab();
        const xaiLocal = getLocalTab();

        expect(xaiGlobal).toHaveAttribute("aria-selected", "true");
        expect(xaiLocal).toHaveAttribute("aria-selected", "false");

        await userEvent.click(xaiLocal);
        expect(xaiLocal).toHaveAttribute("aria-selected", "true");
        expect(xaiGlobal).toHaveAttribute("aria-selected", "false");
    });
});
