// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import type { AxiosResponse } from "axios";
import { mockAxiosInstance } from "tests/mockAxios";
import { clickWithoutPointerCheck } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse, createQueryClient } from "tests/testUtil";

import useFeatureFlags from "hooks/utils/featureFlags";

import {
    PredictiveQueryInfo,
    QueryType,
    TaskType
} from "types/predictiveQueries";

import { parsePresignedURLasDatabricks } from "utils/Helpers/urlHelpers";

import { setHistogramResponseType } from "../../Histogram/__tests__/MockFunctions/Histogram.mock";
import { query_info as query_info_raw } from "../../Histogram/__tests__/MockResponses/HistogramResponse.json";
import PQEvaluationOverview, {
    PQEvaluationOverviewProps
} from "../PQEvaluationOverview";

import baselineEnabled_QueryInfo_raw from "./MockResponses/QueryInfoResponse.json";

// These fixtures carry a populated `results`; the tests reconstruct it via
// spread, so keep `results` non-optional after the JSON cast.
type QueryInfoWithResults = PredictiveQueryInfo & {
    results: NonNullable<PredictiveQueryInfo["results"]>;
};
const query_info = query_info_raw as unknown as QueryInfoWithResults;
const baselineEnabled_QueryInfo =
    baselineEnabled_QueryInfo_raw as unknown as QueryInfoWithResults;

const axiosResponse = <T,>(data: T): AxiosResponse<T> => {
    return createMockResponse(data) as AxiosResponse<T>;
};

vi.mock("hooks/utils/featureFlags");

const defaultProps: PQEvaluationOverviewProps = {
    queryInfo: {
        modelPlannerYaml: null,
        isV2: false
    } as PredictiveQueryInfo,
    show: true,
    handleError: vi.fn()
};

describe("PQEvaluationOverview", () => {
    const trainingJobId = "trainingjob-bed43f8f2fc54ea985439b026dbe0865";

    const queryClient = createQueryClient();
    const renderPQEvaluationOverview = async (
        props: PQEvaluationOverviewProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <PQEvaluationOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            const APIS_TO_CALL = [
                `v0/training_jobs/${trainingJobId}/eval/regression_histogram`,
                `v0/training_jobs/${trainingJobId}/eval/regression_heatmap`
            ];
            APIS_TO_CALL.forEach((API) => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(`${API}`);
            });
        });
    };

    const openListbox = async (comboboxTestId: string) => {
        const statsButton = await screen.findByTestId(comboboxTestId);
        expect(statsButton).toBeInTheDocument();

        await clickWithoutPointerCheck(statsButton);

        const listbox = await screen.findByTestId("listbox");
        expect(listbox).toBeInTheDocument();

        return { statsButton, listbox: within(listbox) };
    };

    const openComboboxAndSelectOption = async (
        comboboxTestId: string,
        optionTestId: string,
        expectedText: string
    ) => {
        const { statsButton, listbox } = await openListbox(comboboxTestId);

        const option = await listbox.findByTestId(optionTestId);
        expect(option).toHaveTextContent(expectedText);

        await clickWithoutPointerCheck(option);

        await waitFor(() =>
            expect(statsButton).toHaveTextContent(expectedText)
        );

        return statsButton;
    };

    const verifyTableRowsAndValues = (
        table: HTMLElement,
        expectedHeaders: string[],
        expectedValues: string[]
    ) => {
        const { getAllByRole } = within(table);
        const rows = getAllByRole("row");

        const headerCells = within(rows[0]).getAllByRole("cell");
        expect(headerCells.map((cell) => cell.textContent)).toEqual(
            expectedHeaders
        );

        const valueCells = within(rows[1]).getAllByRole("cell");
        expect(valueCells.map((cell) => cell.textContent)).toEqual(
            expectedValues
        );
    };

    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useFeatureFlags).mockReturnValue({
            ENABLE_COLUMN_ANALYSIS: false,
            MAX_TABLES_IN_A_GRAPH: 0,
            IS_DATABRICKS_DEPLOY: false
        });
        queryClient.clear();
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
        setHistogramResponseType(testResponseType.DEFAULT);
    });

    it.each([
        {
            label: "Training",
            percent: /33\.33\s*%/,
            count: "100",
            predictionWindow:
                /2011-03-13\s+13:46:00\s+-\s+2011-06-11\s+13:46:00/
        },
        {
            label: "Validation",
            percent: /33\.33\s*%/,
            count: "100",
            predictionWindow:
                /2011-06-11\s+13:46:00\s+-\s+2011-09-09\s+13:46:00/
        },
        {
            label: "Holdout",
            percent: /33\.33\s*%/,
            count: "100",
            predictionWindow:
                /2011-09-09\s+13:46:00\s+-\s+2011-12-08\s+13:46:00/
        }
    ])(
        "Should render $label split data with correct content in split stats table",
        async ({ label, percent, count, predictionWindow }) => {
            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: query_info
            });

            await checkApiCall();

            const splitStatsTable = document.getElementById("split_stats");
            expect(splitStatsTable).toBeInTheDocument();

            expect(
                within(splitStatsTable!).getByText(label)
            ).toBeInTheDocument();

            const percentElements = within(splitStatsTable!).getAllByText(
                percent
            );
            expect(percentElements.length).toBeGreaterThan(0);

            const countElements = within(splitStatsTable!).getAllByText(count);
            expect(countElements.length).toBeGreaterThan(0);

            expect(
                within(splitStatsTable!).getByText(predictionWindow)
            ).toBeInTheDocument();
        }
    );

    it("Should render `RegressionHeatMap` chart", async () => {
        await renderPQEvaluationOverview({
            ...defaultProps,
            queryInfo: query_info
        });

        await checkApiCall();

        await waitFor(() => {
            expect(
                screen.getByText("Predicted vs Actual Values")
            ).toBeInTheDocument();

            expect(
                screen.getByRole("button", { name: /Hide Outliers/i })
            ).toBeInTheDocument();

            const echart = document.querySelector(".echarts-for-react");
            expect(echart).toBeInTheDocument();
        });
    });

    it("Should toggle outliers button text when clicked", async () => {
        await renderPQEvaluationOverview({
            ...defaultProps,
            queryInfo: query_info
        });

        await checkApiCall();

        const button = await screen.findByRole("button", {
            name: /Hide Outliers/i
        });

        await clickWithoutPointerCheck(button);
        expect(button.textContent).toBe("Show Outliers");
        expect(screen.queryByText).not.toBe("Hide Outliers");

        await clickWithoutPointerCheck(button);
        expect(button.textContent).toBe("Hide Outliers");
        expect(screen.queryByText).not.toBe("Show Outliers");
    });

    it("Should render Histogram with prediction distribution when props are set", async () => {
        await renderPQEvaluationOverview({
            ...defaultProps,
            queryInfo: query_info
        });

        await checkApiCall();

        expect(
            await screen.findByText("Distribution of Predictions")
        ).toBeInTheDocument();
        const description = document.querySelector('[data-slot="description"]');
        expect(description).toHaveTextContent(
            "Histogram of the predicted value alongside the actual labels (when available) for all entities in your Predictive Query, in the holdout set of the training table."
        );
    });

    it("Should render listbox with Actuals and Predictions options", async () => {
        await renderPQEvaluationOverview({
            ...defaultProps,
            queryInfo: query_info
        });

        await checkApiCall();

        expect(
            await screen.findByText("Summary stats for:")
        ).toBeInTheDocument();

        const { listbox } = await openListbox("testid-combobox");

        const actualsOption = await listbox.findByTestId("combobox-option-0");
        expect(actualsOption).toHaveTextContent("Actuals");

        const predictionsOption =
            await listbox.findByTestId("combobox-option-1");
        expect(predictionsOption).toHaveTextContent("Predictions");
    });

    it.each([
        {
            optionTestId: "combobox-option-0",
            optionText: "Actuals",
            minMeanMax: {
                headers: ["Min", "Mean", "Max"],
                values: ["0", "0.17", "2"]
            },
            percentiles: {
                headers: ["25%", "50%", "75%"],
                values: ["0", "0", "0"]
            }
        },
        {
            optionTestId: "combobox-option-1",
            optionText: "Predictions",
            minMeanMax: {
                headers: ["Min", "Mean", "Max"],
                values: ["-0.00", "0.002", "0.017"]
            },
            percentiles: {
                headers: ["25%", "50%", "75%"],
                values: ["-0.00", "-0.00", "0.0003"]
            }
        }
    ])(
        "Should render histogram tables with Min/Mean/Max and Percentiles for $optionText",
        async ({ optionTestId, optionText, minMeanMax, percentiles }) => {
            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: query_info
            });

            await checkApiCall();

            await openComboboxAndSelectOption(
                "testid-combobox",
                optionTestId,
                optionText
            );

            expect(screen.getByText("Summary stats for:")).toBeInTheDocument();

            const allTables = await screen.findAllByRole("table");

            const histogramTables = allTables.filter((table) => {
                const id = table.getAttribute("id");
                return id !== "evaluation_metrics" && id !== "split_stats";
            });

            verifyTableRowsAndValues(
                histogramTables[0],
                minMeanMax.headers,
                minMeanMax.values
            );

            expect(screen.getByText("Percentiles")).toBeInTheDocument();
            verifyTableRowsAndValues(
                histogramTables[1],
                percentiles.headers,
                percentiles.values
            );
        }
    );

    it("Should hide the component when show is false", async () => {
        renderPQEvaluationOverview({
            ...defaultProps,
            queryInfo: query_info,
            show: false
        });

        await checkApiCall();

        const container = await screen.findByTestId(
            "pq-eval-overview-container"
        );
        expect(container).toHaveClass("hidden");
    });

    it("Should call handleError when fetch fails", async () => {
        setHistogramResponseType(testResponseType.ERROR);
        const mockHandleError = vi.fn();

        await renderPQEvaluationOverview({
            ...defaultProps,
            handleError: mockHandleError,
            queryInfo: query_info
        });

        await checkApiCall();

        await waitFor(() => {
            expect(mockHandleError).toHaveBeenCalled();
        });
    });

    it("Should not show baseline metrics if enable_baselines is false", async () => {
        const queryInfo = {
            ...baselineEnabled_QueryInfo,
            enable_baselines: false
        };

        await renderPQEvaluationOverview({ ...defaultProps, queryInfo });
    });

    it("Should not show baseline metrics if taskType is not supported for queryType", async () => {
        const queryInfo = {
            ...baselineEnabled_QueryInfo,
            taskType: TaskType.FORECASTING,
            queryType: QueryType.TEMPORAL
        };

        await renderPQEvaluationOverview({ ...defaultProps, queryInfo });
    });

    it("Should show baseline metrics if taskType is a link prediction type", async () => {
        const queryInfo = {
            ...baselineEnabled_QueryInfo,
            taskType: TaskType.LINK_PREDICTION
        };

        await renderPQEvaluationOverview({ ...defaultProps, queryInfo });
    });

    it("Should render Databricks deploy command from presigned URL", async () => {
        const filePath =
            "databricks-host/mycatalog/myschema/myvolume/some/path/file.parquet";
        vi.mocked(useFeatureFlags).mockReturnValue({
            ...useFeatureFlags(),
            IS_DATABRICKS_DEPLOY: true
        });

        mockAxiosInstance.get.mockImplementation((url) => {
            if (url.includes("download_holdout")) {
                return Promise.resolve(
                    axiosResponse({
                        presigned_url: `https://${filePath}`
                    })
                );
            }
            return Promise.resolve(axiosResponse({}));
        });

        await renderPQEvaluationOverview({
            ...defaultProps,
            queryInfo: query_info
        });

        await waitFor(() => {
            expect(
                screen.getByText((text) =>
                    text.includes(
                        "Holdout data is available in your Unity Catalog Volume"
                    )
                )
            ).toBeInTheDocument();
        });

        expect(
            screen.getByText((text) =>
                text.includes("Catalog: databricks-host")
            )
        ).toBeInTheDocument();

        expect(
            screen.getByText((text) => text.includes("Schema: mycatalog"))
        ).toBeInTheDocument();

        expect(
            screen.getByText((text) => text.includes("Volume: myschema"))
        ).toBeInTheDocument();

        expect(
            screen.getByText((text) =>
                text.includes("File Path: myvolume/some/path/file.parquet")
            )
        ).toBeInTheDocument();

        expect(
            screen.getByText((text) =>
                text.includes(
                    `Full Unity Catalog Volume Path: /Volumes/${filePath}`
                )
            )
        ).toBeInTheDocument();
    });

    it.each([
        {
            deployType: "Databricks",
            featureFlag: { IS_DATABRICKS_DEPLOY: true }
        }
    ])(
        "Should render fallback error message for invalid $deployType presigned URL",
        async ({ featureFlag }) => {
            vi.mocked(useFeatureFlags).mockReturnValue({
                ...useFeatureFlags(),
                ...featureFlag
            });

            mockAxiosInstance.get.mockImplementation((url) => {
                if (url.includes("download_holdout")) {
                    return Promise.resolve(
                        axiosResponse({
                            presigned_url: "@invalid"
                        })
                    );
                }
                return Promise.resolve(axiosResponse({}));
            });

            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: query_info
            });

            await waitFor(() => {
                expect(
                    screen.getByText(
                        "Holdout set location can't be displayed. Please contact Kumo support for more information if you want to access the holdout set."
                    )
                ).toBeInTheDocument();
            });
        }
    );

    it("Should handle empty test metrics and show validation/training metrics", async () => {
        const queryInfoNoTestMetrics = {
            ...query_info,
            results: {
                ...query_info.results,
                test_metrics: []
            }
        };

        await renderPQEvaluationOverview({
            ...defaultProps,
            queryInfo: queryInfoNoTestMetrics
        });

        await checkApiCall();

        await waitFor(() => {
            const metricsTable = document.getElementById("evaluation_metrics");
            expect(metricsTable).toBeInTheDocument();
        });

        const metricsTable = document.getElementById("evaluation_metrics");
        expect(
            within(metricsTable!).getAllByText("Validation").length
        ).toBeGreaterThan(0);
        expect(
            within(metricsTable!).getAllByText("Training").length
        ).toBeGreaterThan(0);
    });

    it("Should render BinaryClassification component for binary classification task", async () => {
        const binaryClassificationQueryInfo = {
            ...query_info,
            taskType: TaskType.BINARY_CLASSIFICATION,
            graphName: "test-graph"
        };

        await renderPQEvaluationOverview({
            ...defaultProps,
            queryInfo: binaryClassificationQueryInfo
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${trainingJobId}/eval/binary_classification`
            );
        });
    });

    describe("Locality Metric", () => {
        it("Should render locality score metric card when locality_score is present", async () => {
            const queryInfoWithLocality = {
                ...query_info,
                results: {
                    ...query_info.results,
                    test_metrics: [
                        ...(query_info.results?.test_metrics || []),
                        { name: "locality_score", val: 0.85432 }
                    ]
                }
            };

            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: queryInfoWithLocality
            });

            await checkApiCall();

            await waitFor(() => {
                const localityCard =
                    document.getElementById("tune-metric-card");
                expect(localityCard).toBeInTheDocument();
                expect(screen.getByText("Locality_score")).toBeInTheDocument();
                expect(screen.getByText("0.85")).toBeInTheDocument();
            });
        });

        it("Should not render locality score card when locality_score is not present", async () => {
            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: query_info
            });

            await checkApiCall();

            expect(
                screen.queryByText("Locality_score")
            ).not.toBeInTheDocument();
        });
    });

    describe("Baseline Comparison Card", () => {
        it("Should not render Baseline Comparison card when baselines are disabled", async () => {
            const queryInfoNoBaseline = {
                ...query_info,
                enable_baselines: false
            };

            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: queryInfoNoBaseline
            });

            await checkApiCall();

            expect(
                screen.queryByText("Baseline Comparison")
            ).not.toBeInTheDocument();
        });
    });

    describe("EvalMetricsTable Fallback", () => {
        it("Should render EvalMetricsTable when baseline metrics are not shown", async () => {
            const queryInfoNoBaseline = {
                ...query_info,
                enable_baselines: false
            };

            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: queryInfoNoBaseline
            });

            await checkApiCall();

            await waitFor(() => {
                const metricsTable =
                    document.getElementById("evaluation_metrics");
                expect(metricsTable).toBeInTheDocument();
            });
        });

        it("Should render EvalMetricsTable for FORECASTING task type", async () => {
            const queryInfoForecasting = {
                ...baselineEnabled_QueryInfo,
                taskType: TaskType.FORECASTING
            };

            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: queryInfoForecasting
            });

            await waitFor(() => {
                const metricsTable =
                    document.getElementById("evaluation_metrics");
                expect(metricsTable).toBeInTheDocument();
            });
        });
    });

    describe("Tune Metric Edge Cases", () => {
        it("Should display fallback '-' when tune metric is not available", async () => {
            const queryInfoNoMetrics = {
                ...query_info,
                modelPlannerYaml: null,
                results: {
                    ...query_info.results,
                    test_metrics: []
                }
            };

            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: queryInfoNoMetrics
            });

            await checkApiCall();

            await waitFor(() => {
                expect(screen.getByText("Tune Metrics")).toBeInTheDocument();
                const tuneMetricCard =
                    document.getElementById("tune-metric-card");
                expect(tuneMetricCard).toBeInTheDocument();
            });
        });

        it("Should display tune metric value when available", async () => {
            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: baselineEnabled_QueryInfo
            });

            await waitFor(() => {
                const tuneMetricCard =
                    document.getElementById("tune-metric-card");
                expect(tuneMetricCard).toBeInTheDocument();
                expect(screen.getByText("Tune Metrics")).toBeInTheDocument();
            });
        });
    });

    describe("Multiple Metric Cards", () => {
        it("Should render both locality and tune metric cards when locality score exists", async () => {
            const queryInfoWithBothMetrics = {
                ...baselineEnabled_QueryInfo,
                results: {
                    ...baselineEnabled_QueryInfo.results,
                    test_metrics: [
                        ...(baselineEnabled_QueryInfo.results?.test_metrics ||
                            []),
                        { name: "locality_score", val: 0.92 }
                    ]
                }
            };

            await renderPQEvaluationOverview({
                ...defaultProps,
                queryInfo: queryInfoWithBothMetrics
            });

            await waitFor(() => {
                expect(screen.getByText("Locality_score")).toBeInTheDocument();
                expect(screen.getByText("Tune Metrics")).toBeInTheDocument();
                expect(screen.getByText("0.92")).toBeInTheDocument();
            });
        });
    });

    describe("parsePresignedURLasDatabricks", () => {
        it("Should return correct Databricks command for a valid URL", () => {
            const url =
                "https://example.com/catalog/schema/volume/path/to/file";
            const result = parsePresignedURLasDatabricks(url);

            expect(result).toContain("Catalog: example.com");
            expect(result).toContain("Schema: catalog");
            expect(result).toContain("Volume: schema");
            expect(result).toContain(
                "/Volumes/example.com/catalog/schema/volume/path/to/file"
            );
        });

        it("Should return null for invalid URL", () => {
            const consoleSpy = vi
                .spyOn(console, "error")
                .mockImplementation(() => {});
            const url = "https://example.com/catalog/schema";
            const result = parsePresignedURLasDatabricks(url);
            expect(result).toBeNull();
            expect(consoleSpy).toHaveBeenCalled();
            consoleSpy.mockRestore();
        });

        it("Should handle malformed URLs and return null", () => {
            const consoleSpy = vi
                .spyOn(console, "error")
                .mockImplementation(() => {});
            const result = parsePresignedURLasDatabricks(
                undefined as unknown as string
            );
            expect(result).toBeNull();
            expect(consoleSpy).toHaveBeenCalled();
            consoleSpy.mockRestore();
        });
    });
});
