// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { clickWithoutPointerCheck, typeWithoutDelay } from "tests/testHelpers";
import { createQueryClient } from "tests/testUtil";

import useFeatureFlags from "hooks/utils/featureFlags";

import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";

import PQExplanationOverview, {
    PQExplanationProps
} from "../Overview/PQExplanationOverview";

import binaryQueryInfo from "./MockResponses/QueryInfo.json";

vi.mock("hooks/utils/featureFlags");

const id = "trainingjob-00046000000000000000000000000000";
const defaultProps: PQExplanationProps = {
    queryInfo: binaryQueryInfo as unknown as PredictiveQueryInfo,
    show: true
};

describe("PQExplanationOverview", () => {
    const queryClient = createQueryClient();
    const renderPQExplanation = async (props: PQExplanationProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <PQExplanationOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async (id: string) => {
        await waitFor(() => {
            const APIS_TO_CALL = [
                `v0/training_jobs/${id}/eval/available_xai_entities`,
                `v0/training_jobs/${id}/eval/xai`
            ];
            APIS_TO_CALL.forEach((API) => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(API);
            });
        });
    };

    const expectBasicTabsAndPlaceholder = () => {
        const xaiGlobal = screen.getByRole("tab", {
            name: "Global column explanations"
        });
        const xaiLocal = screen.getByRole("tab", {
            name: "Local subgraph explanation"
        });
        return { xaiGlobal, xaiLocal };
    };

    const navigateToLocalTab = async () => {
        await clickWithoutPointerCheck(
            screen.getByText("Local subgraph explanation")
        );
    };

    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useFeatureFlags).mockReturnValue({
            ENABLE_COLUMN_ANALYSIS: true,
            MAX_TABLES_IN_A_GRAPH: 0,
            IS_DATABRICKS_DEPLOY: false
        });
        queryClient.clear();
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should render `PQExplanation` when queryInfo is empty or null", async () => {
        await renderPQExplanation({
            queryInfo: {} as PredictiveQueryInfo,
            show: false
        });

        expectBasicTabsAndPlaceholder();
    });

    it("Should display all the content under the `Global` when all the props are set", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        expectBasicTabsAndPlaceholder();

        //Global
        const columnName = "FINANCE_ORDERS.AMOUNT";
        await waitFor(() =>
            expect(
                screen.getByTestId(`column-${columnName} (1 hop)`)
            ).toBeInTheDocument()
        );
        await waitFor(() =>
            expect(screen.getAllByText(`${columnName} (1 hop)`).length).toBe(1)
        );

        expect(await screen.findByText("Hops")).toBeInTheDocument();
        expect(await screen.findByText("Type")).toBeInTheDocument();
        expect(
            await screen.findByText("Variation of prediction")
        ).toBeInTheDocument();

        await waitFor(() =>
            expect(screen.getByTestId("bar-chart")).toBeInTheDocument()
        );
    });

    it("Should filter the columns based on the `Search`", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        const searchInput =
            await screen.findByPlaceholderText("Search columns");
        expect(screen.getByTestId("xai-columns-search")).toBeInTheDocument();

        const input = "FINANCE_TRANS.BALANCE";
        await typeWithoutDelay(searchInput, input);

        await waitFor(() =>
            expect(
                screen.getByTestId(`column-${input} (1 hop)`)
            ).toBeInTheDocument()
        );
        expect(screen.getAllByText(`${input} (1 hop)`).length).toBe(1);

        const nonSearchInput = "FINANCE_ORDERS.AMOUNT";
        expect(screen.queryByText(nonSearchInput)).not.toBeInTheDocument();
    });

    it("Should display `No columns` for non-existing `Search`", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        const searchInput =
            await screen.findByPlaceholderText("Search columns");
        expect(screen.getByTestId("xai-columns-search")).toBeInTheDocument();

        const input = "302000#5";
        await typeWithoutDelay(searchInput, input);

        await waitFor(() =>
            expect(screen.getByText("No columns")).toBeInTheDocument()
        );
    });

    it("Should display all the content under the `Local` when all the props are set", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        await navigateToLocalTab();

        //Local
        const categoryDropdown = screen.getByTestId("testid-category-dropdown");
        expect(categoryDropdown).toHaveTextContent("Random");

        expect(
            screen.getByRole("button", { name: /Previous/i })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Next/i })
        ).toBeInTheDocument();
        expect(screen.getByText("Sample details")).toBeInTheDocument();
    });

    it("Should display various options under the category dropdown", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        await navigateToLocalTab();

        const categoryDropdown = screen.getByTestId("testid-category-dropdown");
        expect(categoryDropdown).toHaveTextContent("Random");
        await clickWithoutPointerCheck(categoryDropdown);
        const options = [
            { testId: "category-dropdown-option-0", label: "Random" },
            { testId: "category-dropdown-option-1", label: "Unsure" },
            { testId: "category-dropdown-option-2", label: "True Positive" },
            { testId: "category-dropdown-option-3", label: "True Negative" },
            { testId: "category-dropdown-option-4", label: "False Positive" },
            { testId: "category-dropdown-option-5", label: "False Negative" },
            { testId: "category-dropdown-option-6", label: "Old Entity" },
            { testId: "category-dropdown-option-7", label: "Recent Entity" }
        ];

        await screen.findByRole("menuitem", { name: options[0].label });
        for (const { label } of options) {
            expect(
                screen.getByRole("menuitem", { name: label })
            ).toBeInTheDocument();
        }
    });

    it("Should change the sample id when `Next` or `Previous` button is clicked", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        await navigateToLocalTab();

        const previousButton = screen.getByRole("button", {
            name: /Previous/i
        });
        const nextButton = screen.getByRole("button", { name: /Next/i });

        const initialId = "2026";
        expect(screen.getByText(initialId)).toBeInTheDocument();

        //Next button click
        await clickWithoutPointerCheck(nextButton);
        const nextId = "1228";
        expect(screen.getByText(nextId)).toBeInTheDocument();
        expect(screen.queryByText(initialId)).not.toBeInTheDocument();

        //Previous button click
        await clickWithoutPointerCheck(previousButton);
        expect(screen.getByText(initialId)).toBeInTheDocument();
        expect(screen.queryByText(nextId)).not.toBeInTheDocument();
    });

    it("Should display content under the Sample details", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        await navigateToLocalTab();

        expect(screen.getByText("Sample details")).toBeInTheDocument();
        expect(screen.getByText("Output")).toBeInTheDocument();
        expect(screen.getByText("Prediction")).toBeInTheDocument();
        expect(screen.getByText("Ground Truth")).toBeInTheDocument();
        expect(screen.getByText("Anchor Time")).toBeInTheDocument();
    });

    it("Should display subgraph,subgraph table, and top 3 columns table under the `Local`", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        await navigateToLocalTab();

        //Subgraph
        expect(await screen.findByText("Subgraph")).toBeInTheDocument();
        expect(
            screen.getByText(/Press enter or space to select a node./i)
        ).toBeInTheDocument();
        expect(screen.getByTestId("rf__controls")).toBeInTheDocument();
        expect(screen.getByTestId("rf__wrapper")).toBeInTheDocument();

        //Subgraph Summary table
        const tabs = screen.getAllByRole("tab");

        expect(tabs[2]).toHaveTextContent("Subgraph summary");
        await clickWithoutPointerCheck(tabs[2]);
        const subgraphTableHeader = screen.getByTestId("table-header");
        expect(subgraphTableHeader).toBeInTheDocument();
        expect(subgraphTableHeader).toHaveTextContent(
            "PropertyCurrent EntityGlobal"
        );

        //Top 3 columns table
        const top3TableTab = tabs[3];
        await clickWithoutPointerCheck(top3TableTab);

        const top3TableHeader = screen.getByTestId("table-header");
        expect(top3TableHeader).toBeInTheDocument();
        expect(top3TableHeader).toHaveTextContent("TableColumnValueMode");
    });

    it.each([
        {
            taskType: TaskType.MULTILABEL_RANKING,
            expectedText:
                /Column Analysis is not available for MULTILABEL RANKING tasks. For more information, please contact kumo support./i
        },
        {
            taskType: TaskType.LINK_PREDICTION,
            expectedText:
                /Column Analysis is not available for LINK PREDICTION tasks. For more information, please contact kumo support./i
        }
    ])(
        "Should display `Column Analysis is not available` for taskType $taskType",
        async ({ taskType, expectedText }) => {
            await renderPQExplanation({
                ...defaultProps,
                queryInfo: {
                    "id": "trainingjob-test",
                    "trainingStatus": "DONE",
                    "taskType": taskType
                } as unknown as PredictiveQueryInfo
            });

            expectBasicTabsAndPlaceholder();

            await waitFor(() => {
                expect(screen.getByText(expectedText)).toBeInTheDocument();
            });
        }
    );

    it("Should not display column analysis when feature flag is disabled", async () => {
        vi.mocked(useFeatureFlags).mockReturnValue({
            ...useFeatureFlags(),
            ENABLE_COLUMN_ANALYSIS: false
        });

        await renderPQExplanation(defaultProps);

        await waitFor(() => {
            expect(
                screen.getByText(
                    /Column Analysis is not available in this version of Kumo/i
                )
            ).toBeInTheDocument();
        });
    });

    it("Should maintain state when switching between Global and Local tabs", async () => {
        await renderPQExplanation(defaultProps);
        await checkApiCall(id);

        const { xaiGlobal } = expectBasicTabsAndPlaceholder();
        const searchInput =
            await screen.findByPlaceholderText("Search columns");
        await typeWithoutDelay(searchInput, "FINANCE");

        await clickWithoutPointerCheck(
            screen.getByRole("tab", { name: "Local subgraph explanation" })
        );

        await clickWithoutPointerCheck(xaiGlobal);
        expect(searchInput).toHaveValue("FINANCE");
    });

    it("Should hide tab navigation and show local content directly for TLP tasks", async () => {
        await renderPQExplanation({
            ...defaultProps,
            queryInfo: {
                "id": "trainingjob-test",
                "trainingStatus": "DONE",
                "taskType": TaskType.TEMPORAL_LINK_PREDICTION,
                "trainingJobId": "trainingjob-test"
            } as unknown as PredictiveQueryInfo
        });

        // Tab switcher should not be rendered for TLP
        expect(
            screen.queryByRole("tab", { name: "Global column explanations" })
        ).not.toBeInTheDocument();
        expect(
            screen.queryByRole("tab", { name: "Local subgraph explanation" })
        ).not.toBeInTheDocument();

        // "Local Subgraph Explanation" title should be shown as standalone heading
        expect(
            screen.getByText("Local Subgraph Explanation")
        ).toBeInTheDocument();
    });

    it("Should render Sample evaluation tab as default for TLP tasks", async () => {
        await renderPQExplanation({
            ...defaultProps,
            queryInfo: {
                "id": "trainingjob-test",
                "trainingStatus": "DONE",
                "taskType": TaskType.TEMPORAL_LINK_PREDICTION,
                "trainingJobId": "trainingjob-test"
            } as unknown as PredictiveQueryInfo
        });

        await waitFor(() => {
            expect(
                screen.getByRole("tab", { name: /Sample evaluation/i })
            ).toBeInTheDocument();
        });

        // Sample evaluation tab should be active by default
        const sampleEvalTab = screen.getByRole("tab", {
            name: /Sample evaluation/i
        });
        expect(sampleEvalTab).toHaveAttribute("data-state", "active");

        // Other tabs should be present
        expect(
            screen.getByRole("tab", { name: /Top 3 columns/i })
        ).toBeInTheDocument();

        // Sample details card heading should be present
        expect(screen.getByText("Sample details")).toBeInTheDocument();
    });
});
