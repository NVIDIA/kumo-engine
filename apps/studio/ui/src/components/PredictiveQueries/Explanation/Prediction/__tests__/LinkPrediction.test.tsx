// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HistoricalItems } from "gen/v0/types";
import { createQueryClient } from "tests/testUtil";
import { vi } from "vitest";

import { LinkPredictionEntityOutput } from "types/explorer";
import { TaskType } from "types/predictiveQueries";

import { staticLpEntityOutput } from "../../__tests__/MockResponses/EntityOutputResponses";
import { staticLpHistoricalItems } from "../../__tests__/MockResponses/HistoricalItemsResponses";
import { staticLpTarget } from "../../__tests__/MockResponses/TargetResponses";
import LinkPrediction from "../LinkPrediction";

interface LinkPredictionTestProps {
    entityOutput: LinkPredictionEntityOutput;
    historicalItems: HistoricalItems;
    isDataLoading: boolean;
    showGroundTruth?: boolean;
    selectedColumns?: Array<string>;
    setSelectedColumns?: (columns: Array<string>) => void;
}

const defaultProps: LinkPredictionTestProps = {
    entityOutput: {
        task_type: TaskType.STATIC_LINK_PREDICTION,
        prediction: staticLpEntityOutput,
        target: staticLpTarget
    } as LinkPredictionEntityOutput,
    historicalItems: staticLpHistoricalItems,
    isDataLoading: false,
    showGroundTruth: true
};

describe("LinkPrediction", () => {
    const renderLinkPrediction = async (
        props: LinkPredictionTestProps = defaultProps
    ) => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <LinkPrediction {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const getChip = (colName: string) => screen.getByTestId(`chip-${colName}`);

    const getRemoveChipButton = (colName: string) =>
        screen.getByTestId(`remove-chip-${colName}`);

    const hasColumn = (columnName: string) => {
        const columnHeaders = screen.getAllByRole("columnheader");
        return columnHeaders.some((header) =>
            header.textContent?.includes(columnName)
        );
    };

    it("Should render all sections of LinkPrediction with expected data", async () => {
        await renderLinkPrediction();

        expect(screen.getByText("Columns")).toBeInTheDocument();
        // customer_id and FN are the first 2 columns selected by default
        expect(getChip("customer_id")).toBeInTheDocument();
        expect(getChip("FN")).toBeInTheDocument();

        expect(
            screen.getByText("Historical Items (1 items)")
        ).toBeInTheDocument();
        expect(
            screen.getByText(staticLpHistoricalItems.nodes[0].id)
        ).toBeInTheDocument();

        expect(screen.getByText("Ground Truth (1 items)")).toBeInTheDocument();
        expect(
            screen.getByText(staticLpEntityOutput[0].id)
        ).toBeInTheDocument();

        expect(screen.getByText("Predictions (10 items)")).toBeInTheDocument();
        expect(
            screen.getAllByText(staticLpTarget[0].id).length
        ).toBeGreaterThan(0);
    });

    it("Should not display Ground Truth when showGroundTruth is false", async () => {
        await renderLinkPrediction({ ...defaultProps, showGroundTruth: false });

        expect(
            screen.getByText("Historical Items (1 items)")
        ).toBeInTheDocument();
        expect(screen.getByText("Predictions (10 items)")).toBeInTheDocument();
        expect(
            screen.queryByText("Ground Truth (1 items)")
        ).not.toBeInTheDocument();
    });

    it("Should filter Predictions based on search input", async () => {
        await renderLinkPrediction();

        const searchInputs = screen.getAllByPlaceholderText("Search");
        const predictionsSearchInput = searchInputs[2];
        const inputToSearch = staticLpEntityOutput[7].id;
        const otherPredictionId = staticLpEntityOutput[4].id;

        expect(screen.getByText(otherPredictionId)).toBeInTheDocument();

        await userEvent.click(predictionsSearchInput);
        await userEvent.type(predictionsSearchInput, inputToSearch);

        expect(screen.getByText(inputToSearch)).toBeInTheDocument();
        expect(screen.queryByText(otherPredictionId)).not.toBeInTheDocument();

        await userEvent.clear(predictionsSearchInput);
        expect(screen.queryByText(otherPredictionId)).toBeInTheDocument();
    });

    it("Should display additional columns when added via dropdown", async () => {
        await renderLinkPrediction();

        expect(screen.queryByTestId("chip-Active")).not.toBeInTheDocument();

        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("Active"));

        expect(getChip("Active")).toBeInTheDocument();
        expect(hasColumn("Active")).toBe(true);
    });

    it("Should render nothing if no columns exist in prediction data", async () => {
        const propsWithNoCols = {
            ...defaultProps,
            entityOutput: {
                ...defaultProps.entityOutput,
                prediction: [{ id: "test", cols: [], id_name: "" }]
            }
        };
        await renderLinkPrediction(propsWithNoCols);

        expect(screen.queryByText("Columns")).not.toBeInTheDocument();
    });

    it("Should filter Historical Items based on search input", async () => {
        await renderLinkPrediction();

        const searchInputs = screen.getAllByPlaceholderText("Search");
        const historicalSearchInput = searchInputs[0];
        const historicalItemId = staticLpHistoricalItems.nodes[0].id;

        await userEvent.click(historicalSearchInput);
        await userEvent.type(historicalSearchInput, historicalItemId);

        expect(screen.getByText(historicalItemId)).toBeInTheDocument();
    });

    it("Should filter Ground Truth based on search input", async () => {
        await renderLinkPrediction();

        const searchInputs = screen.getAllByPlaceholderText("Search");
        const groundTruthSearchInput = searchInputs[1];
        const groundTruthId = staticLpEntityOutput[0].id;

        await userEvent.click(groundTruthSearchInput);
        await userEvent.type(groundTruthSearchInput, groundTruthId);

        expect(screen.getByText(groundTruthId)).toBeInTheDocument();
    });

    it("Should display multiple columns when added via dropdown", async () => {
        await renderLinkPrediction();

        // customer_id and FN are default chips; add age and Active via dropdown
        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("age"));

        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("Active"));

        expect(getChip("age")).toBeInTheDocument();
        expect(getChip("Active")).toBeInTheDocument();
        expect(hasColumn("age")).toBe(true);
        expect(hasColumn("Active")).toBe(true);
    });

    it("Should remove columns when chip × button is clicked", async () => {
        await renderLinkPrediction();

        // FN is a default chip; add age via dropdown
        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("age"));

        expect(getChip("age")).toBeInTheDocument();
        expect(getChip("FN")).toBeInTheDocument();
        expect(hasColumn("age")).toBe(true);
        expect(hasColumn("FN")).toBe(true);

        // Remove age via chip × button
        await userEvent.click(getRemoveChipButton("age"));

        await waitFor(() => {
            expect(screen.queryByTestId("chip-age")).not.toBeInTheDocument();
        });
        expect(getChip("FN")).toBeInTheDocument();
        expect(hasColumn("age")).toBe(false);
        expect(hasColumn("FN")).toBe(true);
    });

    it("Should display loading indicator in section headers when isDataLoading is true", async () => {
        await renderLinkPrediction({ ...defaultProps, isDataLoading: true });

        const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
        expect(skeletons.length).toBe(3);

        expect(
            screen.getByText("Historical Items (... items)")
        ).toBeInTheDocument();
        expect(screen.getByText("Predictions (... items)")).toBeInTheDocument();
        expect(
            screen.getByText("Ground Truth (... items)")
        ).toBeInTheDocument();
    });

    it("Should render nothing when no predictions are available", async () => {
        await renderLinkPrediction({
            ...defaultProps,
            entityOutput: {
                ...defaultProps.entityOutput,
                prediction: []
            }
        });

        expect(screen.queryByText("Columns")).not.toBeInTheDocument();
    });

    it("Should render image columns when stype is image", async () => {
        const imageColumn = {
            name: "product_image",
            value: "https://example.com/image.jpg",
            stype: "image",
            grad_score: undefined
        };

        const entityOutputWithImage = {
            ...defaultProps.entityOutput,
            prediction: [
                {
                    ...staticLpEntityOutput[0],
                    cols: [...staticLpEntityOutput[0].cols, imageColumn]
                }
            ]
        };

        await renderLinkPrediction({
            ...defaultProps,
            entityOutput:
                entityOutputWithImage as typeof defaultProps.entityOutput
        });

        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("product_image"));

        const images = screen.getAllByRole("img");
        expect(images.length).toBeGreaterThan(0);
    });

    it("Should highlight cells with emerald-50 background when item is in both prediction and target", async () => {
        await renderLinkPrediction();

        const tables = screen.getAllByRole("table");
        const predictionsTable = tables[tables.length - 1];

        const highlightedCells = predictionsTable.querySelectorAll(
            '[class*="bg-emerald-50"]'
        );
        expect(highlightedCells.length).toBeGreaterThan(0);
    });

    it("Should handle empty historical items", async () => {
        await renderLinkPrediction({
            ...defaultProps,
            historicalItems: { nodes: [] }
        });

        expect(
            screen.getByText(/No historical items data available/i)
        ).toBeInTheDocument();
    });

    it("Should have exactly 10 items visible on first page when there are 10 predictions", async () => {
        await renderLinkPrediction();

        const tables = screen.getAllByRole("table");
        const predictionsTable = tables[tables.length - 1];

        const rows = predictionsTable.querySelectorAll("tbody tr");
        expect(rows.length).toBe(10);
    });

    it("Should handle entity output without id_name field", async () => {
        const entityOutputWithoutIdName = {
            ...defaultProps.entityOutput,
            prediction: [
                {
                    id: "test-id",
                    cols: [
                        {
                            name: "test_col",
                            value: "test_value",
                            stype: "categorical",
                            grad_score: undefined
                        }
                    ],
                    id_name: ""
                }
            ]
        };

        await renderLinkPrediction({
            ...defaultProps,
            entityOutput:
                entityOutputWithoutIdName as typeof defaultProps.entityOutput,
            historicalItems: {
                nodes: [
                    {
                        id: "hist-id",
                        cols: [
                            {
                                name: "test_col",
                                value: "hist_value",
                                stype: "categorical",
                                grad_score: undefined
                            }
                        ],
                        id_name: ""
                    }
                ]
            }
        });

        expect(screen.getByText("Columns")).toBeInTheDocument();
        const columnHeaders = screen.getAllByRole("columnheader");
        expect(columnHeaders.length).toBeGreaterThan(0);
    });

    it("Should handle columns without name property", async () => {
        const entityOutputWithNullColumn = {
            ...defaultProps.entityOutput,
            prediction: [
                {
                    id: "test-id",
                    cols: [
                        {
                            name: "valid_col",
                            value: "test_value",
                            stype: "categorical",
                            grad_score: undefined
                        },
                        {
                            name: null,
                            value: "invalid",
                            stype: "categorical",
                            grad_score: undefined
                        },
                        {
                            name: "extra_col",
                            value: "extra_value",
                            stype: "categorical",
                            grad_score: undefined
                        }
                    ],
                    id_name: "test_id"
                }
            ]
        };

        await renderLinkPrediction({
            ...defaultProps,
            entityOutput:
                entityOutputWithNullColumn as typeof defaultProps.entityOutput
        });

        expect(screen.getByText("Columns")).toBeInTheDocument();
        // test_id and valid_col are the default chips; verify component handled the null column
        expect(getChip("test_id")).toBeInTheDocument();
        expect(getChip("valid_col")).toBeInTheDocument();

        // Add extra_col via dropdown
        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("extra_col"));

        expect(hasColumn("extra_col")).toBe(true);
    });

    it("Should not re-transform entity output when already transformed", async () => {
        const alreadyTransformedOutput = {
            ...defaultProps.entityOutput,
            prediction: [
                {
                    ...staticLpEntityOutput[0],
                    cols: [
                        ...staticLpEntityOutput[0].cols,
                        {
                            name: "in_ground_truth_or_target",
                            value: true,
                            stype: "boolean",
                            grad_score: undefined
                        }
                    ]
                }
            ]
        };

        await renderLinkPrediction({
            ...defaultProps,
            entityOutput:
                alreadyTransformedOutput as typeof defaultProps.entityOutput
        });

        expect(screen.getByText("Columns")).toBeInTheDocument();
    });

    it("Should use external selectedColumns state when provided", async () => {
        const externalSelectedColumns = ["customer_id", "FN", "Active"];
        const setSelectedColumns = vi.fn();

        await renderLinkPrediction({
            ...defaultProps,
            selectedColumns: externalSelectedColumns,
            setSelectedColumns
        });

        expect(getChip("customer_id")).toBeInTheDocument();
        expect(getChip("FN")).toBeInTheDocument();
        expect(getChip("Active")).toBeInTheDocument();

        expect(hasColumn("FN")).toBe(true);
        expect(hasColumn("Active")).toBe(true);
    });

    it("Should call external setSelectedColumns when column is added", async () => {
        const externalSelectedColumns = ["customer_id"];
        const setSelectedColumns = vi.fn();

        await renderLinkPrediction({
            ...defaultProps,
            selectedColumns: externalSelectedColumns,
            setSelectedColumns
        });

        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("FN"));

        expect(setSelectedColumns).toHaveBeenCalledWith(
            expect.arrayContaining(["customer_id", "FN"])
        );
    });

    it("Should call external setSelectedColumns when chip is removed", async () => {
        const externalSelectedColumns = ["customer_id", "FN"];
        const setSelectedColumns = vi.fn();

        await renderLinkPrediction({
            ...defaultProps,
            selectedColumns: externalSelectedColumns,
            setSelectedColumns
        });

        expect(getChip("FN")).toBeInTheDocument();

        await userEvent.click(getRemoveChipButton("FN"));

        expect(setSelectedColumns).toHaveBeenCalledWith(["customer_id"]);
    });

    it("Should preserve chip state across re-renders with external state", async () => {
        const externalSelectedColumns = ["customer_id", "FN", "age"];
        const setSelectedColumns = vi.fn();
        const queryClient = createQueryClient();

        const { rerender } = render(
            <MemoryRouter>
                <QueryClientProvider client={queryClient}>
                    <LinkPrediction
                        {...defaultProps}
                        selectedColumns={externalSelectedColumns}
                        setSelectedColumns={setSelectedColumns}
                    />
                </QueryClientProvider>
            </MemoryRouter>
        );

        expect(getChip("FN")).toBeInTheDocument();
        expect(getChip("age")).toBeInTheDocument();

        await act(async () => {
            rerender(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <LinkPrediction
                            {...defaultProps}
                            selectedColumns={externalSelectedColumns}
                            setSelectedColumns={setSelectedColumns}
                        />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });

        expect(getChip("FN")).toBeInTheDocument();
        expect(getChip("age")).toBeInTheDocument();
    });

    it("Should render per-table subtitles", async () => {
        await renderLinkPrediction();

        expect(
            screen.getByText(
                "Items this entity interacted with before the prediction window"
            )
        ).toBeInTheDocument();
        expect(
            screen.getByText(
                "Items this entity actually interacted with during the test window"
            )
        ).toBeInTheDocument();
        expect(
            screen.getByText(
                "Items the model predicted this entity would interact with"
            )
        ).toBeInTheDocument();
    });
});
